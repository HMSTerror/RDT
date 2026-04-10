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
: "${BATCH_SIZE:=16}"
: "${NUM_INFERENCE_STEPS:=50}"
: "${TOPK:=5,10,20}"
: "${EXCLUDE_HISTORY_ITEMS:=1}"
: "${GROUP_STRATEGY:=equal_items}"
: "${FREQUENCY_SOURCE_SPLIT:=train}"
: "${POPULARITY_PENALTY:=}"
: "${POPULARITY_PENALTY_SOURCE_SPLIT:=}"
: "${MAX_EVAL_BATCHES:=0}"
: "${PRINT_EVERY:=20}"
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
  python scripts/eval_genrec_hybrid_diffusion.py
  --config_path "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT}"
  --split "${SPLIT}"
  --batch_size "${BATCH_SIZE}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --topk "${TOPK}"
  --group_strategy "${GROUP_STRATEGY}"
  --frequency_source_split "${FREQUENCY_SOURCE_SPLIT}"
  --max_eval_batches "${MAX_EVAL_BATCHES}"
  --print_every "${PRINT_EVERY}"
  --save_json "${JSON_PATH}"
  --save_jsonl "${JSONL_PATH}"
  --save_plot "${PLOT_PATH}"
)

if [[ "${EXCLUDE_HISTORY_ITEMS}" == "1" ]]; then
  CMD+=(--exclude_history_items)
fi

if [[ -n "${POPULARITY_PENALTY}" ]]; then
  CMD+=(--popularity_penalty "${POPULARITY_PENALTY}")
fi

if [[ -n "${POPULARITY_PENALTY_SOURCE_SPLIT}" ]]; then
  CMD+=(--popularity_penalty_source_split "${POPULARITY_PENALTY_SOURCE_SPLIT}")
fi

if [[ -n "${OCCLUDE_MODALITIES}" ]]; then
  CMD+=(--occlude_modalities "${OCCLUDE_MODALITIES}")
fi

echo "========== Hybrid Diffusion Evaluation =========="
echo "checkpoint: ${CHECKPOINT}"
echo "split     : ${SPLIT}"
echo "topk      : ${TOPK}"
echo "pop_pen   : ${POPULARITY_PENALTY:-config-default}"
echo "occlude   : ${OCCLUDE_MODALITIES:-none}"
echo "save_json : ${JSON_PATH}"
echo "save_plot : ${PLOT_PATH}"
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/genrec_hybrid_eval_${SPLIT}_${STAMP}.log"

echo "========== Evaluation done =========="
echo "metrics_json = ${JSON_PATH}"
echo "pred_jsonl   = ${JSONL_PATH}"
echo "plot_png     = ${PLOT_PATH}"
