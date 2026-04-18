#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${SOURCE_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_text.npy}"
: "${TARGET_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_cf.npy}"
: "${OUTPUT_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_text_to_cf.npy}"
: "${SOURCE_NAME:=source}"
: "${TARGET_NAME:=cf}"
: "${HIDDEN_DIM:=256}"
: "${NUM_HIDDEN_LAYERS:=2}"
: "${DROPOUT:=0.05}"
: "${BATCH_SIZE:=1024}"
: "${EPOCHS:=100}"
: "${LEARNING_RATE:=1e-3}"
: "${WEIGHT_DECAY:=1e-4}"
: "${VAL_RATIO:=0.1}"
: "${PATIENCE:=10}"
: "${MSE_WEIGHT:=1.0}"
: "${COSINE_WEIGHT:=1.0}"
: "${CONTRASTIVE_WEIGHT:=0.1}"
: "${CONTRASTIVE_TEMPERATURE:=0.07}"
: "${DEVICE:=}"
: "${SEED:=0}"

export PYTHONUNBUFFERED=1

args=(
  scripts/align_item_embeddings_to_target.py
  --source-path "${SOURCE_PATH}"
  --target-path "${TARGET_PATH}"
  --output-path "${OUTPUT_PATH}"
  --source-name "${SOURCE_NAME}"
  --target-name "${TARGET_NAME}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-hidden-layers "${NUM_HIDDEN_LAYERS}"
  --dropout "${DROPOUT}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --val-ratio "${VAL_RATIO}"
  --patience "${PATIENCE}"
  --mse-weight "${MSE_WEIGHT}"
  --cosine-weight "${COSINE_WEIGHT}"
  --contrastive-weight "${CONTRASTIVE_WEIGHT}"
  --contrastive-temperature "${CONTRASTIVE_TEMPERATURE}"
  --seed "${SEED}"
  --overwrite
)

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

python "${args[@]}" "$@"
