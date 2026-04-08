#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${TEXT_MODEL_NAME_OR_PATH:?Set TEXT_MODEL_NAME_OR_PATH to your local T5 checkpoint path}"
: "${VISION_MODEL_NAME_OR_PATH:=google/siglip-base-patch16-224}"
: "${TOKENIZED_ROOT:=data/Amazon_Music_And_Instruments/tokenized_semantic_ids}"
: "${SEMANTIC_ID_ROOT:=data/Amazon_Music_And_Instruments/semantic_ids_pq}"
: "${BUFFER_ROOT:=buffer/amazon_music_real}"
: "${OUTPUT_DIR:=checkpoints/genrec_dit_amazon}"
: "${CONFIG_PATH:=configs/genrec_dit_amazon.yaml}"
: "${NUM_PROCESSES:=2}"
: "${MIXED_PRECISION:=bf16}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${EVAL_BATCH_SIZE:=16}"
: "${MAX_TRAIN_STEPS:=10000}"
: "${LOG_DIR:=logs}"
: "${ENABLE_IMAGE:=1}"
: "${ENABLE_CF:=1}"
: "${CF_METHOD:=item2vec}"

export TEXT_MODEL_NAME_OR_PATH
export VISION_MODEL_NAME_OR_PATH
export TOKENIZED_ROOT
export SEMANTIC_ID_ROOT
export BUFFER_ROOT
export ENABLE_IMAGE
export ENABLE_CF
export CF_METHOD

mkdir -p "${LOG_DIR}"

echo "========== [1/3] prepare semantic IDs + tokenized samples =========="
bash scripts/prepare_genrec_semantic_ids.sh

echo "========== [2/3] train GenRec DiT =========="
accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  scripts/train_genrec_dit.py \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --max_train_steps "${MAX_TRAIN_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}"

echo "========== [3/3] grouped test evaluation =========="
python scripts/eval_genrec_dit.py \
  --config_path "${CONFIG_PATH}" \
  --checkpoint "${OUTPUT_DIR}/final" \
  --split test \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --topk 5,10,20 \
  --exclude_history_items \
  --group_strategy equal_items \
  --frequency_source_split train \
  --mixed_precision "${MIXED_PRECISION}" \
  --save_json "${LOG_DIR}/genrec_test_grouped_metrics.json" \
  --save_jsonl "${LOG_DIR}/genrec_test_grouped_predictions.jsonl" \
  --save_plot "${LOG_DIR}/genrec_test_grouped_metrics.png"

echo "========== pipeline finished =========="
