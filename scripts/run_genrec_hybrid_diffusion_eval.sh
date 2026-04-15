#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon.yaml}"
: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_50k}"
: "${CHECKPOINT:=}"
: "${SPLIT:=test}"
: "${NUM_PROCESSES:=1}"
: "${MIXED_PRECISION:=no}"
: "${BATCH_SIZE:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${TOPK:=5,10,20}"
: "${EXCLUDE_HISTORY_ITEMS:=1}"
: "${GROUP_STRATEGY:=equal_items}"
: "${FREQUENCY_SOURCE_SPLIT:=train}"
: "${MAX_EVAL_BATCHES:=0}"
: "${PRINT_EVERY:=20}"
: "${EVAL_SEED:=42}"
: "${OCCLUDE_MODALITIES:=}"
: "${LOG_DIR:=logs}"

mkdir -p "${LOG_DIR}"

if [[ -z "${CHECKPOINT}" ]]; then
  if [[ -f "${OUTPUT_DIR}/final/pytorch_model.bin" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/final"
  elif [[ -f "${OUTPUT_DIR}/checkpoint-50000/pytorch_model.bin" ]]; then
    CHECKPOINT="${OUTPUT_DIR}/checkpoint-50000"
  else
    echo "No checkpoint found under ${OUTPUT_DIR}. Set CHECKPOINT manually."
    exit 1
  fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
JSON_PATH="${LOG_DIR}/genrec_hybrid_${SPLIT}_grouped_metrics_${STAMP}.json"
JSONL_PATH="${LOG_DIR}/genrec_hybrid_${SPLIT}_grouped_predictions_${STAMP}.jsonl"
PLOT_PATH="${LOG_DIR}/genrec_hybrid_${SPLIT}_grouped_metrics_${STAMP}.png"

CMD=(
  accelerate launch
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision "${MIXED_PRECISION}"
  scripts/eval_genrec_hybrid_diffusion.py
  --config_path "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --mixed_precision "${MIXED_PRECISION}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --topk "${TOPK}"
  --group_strategy "${GROUP_STRATEGY}"
  --frequency_source_split "${FREQUENCY_SOURCE_SPLIT}"
  --max_eval_batches "${MAX_EVAL_BATCHES}"
  --print_every "${PRINT_EVERY}"
  --seed "${EVAL_SEED}"
  --save_json "${JSON_PATH}"
  --save_jsonl "${JSONL_PATH}"
  --save_plot "${PLOT_PATH}"
)

if [[ "${EXCLUDE_HISTORY_ITEMS}" == "1" ]]; then
  CMD+=(--exclude_history_items)
fi

if [[ -n "${OCCLUDE_MODALITIES}" ]]; then
  CMD+=(--occlude_modalities "${OCCLUDE_MODALITIES}")
fi

echo "========== Hybrid Diffusion Evaluation =========="
echo "checkpoint: ${CHECKPOINT}"
echo "split     : ${SPLIT}"
echo "topk      : ${TOPK}"
echo "num_proc  : ${NUM_PROCESSES}"
echo "precision : ${MIXED_PRECISION}"
echo "eval_seed : ${EVAL_SEED}"
echo "occlude   : ${OCCLUDE_MODALITIES:-none}"
echo "save_json : ${JSON_PATH}"
echo "save_plot : ${PLOT_PATH}"
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/genrec_hybrid_eval_${SPLIT}_${STAMP}.log"

echo "========== Evaluation done =========="
echo "metrics_json = ${JSON_PATH}"
echo "pred_jsonl   = ${JSONL_PATH}"
echo "plot_png     = ${PLOT_PATH}"
