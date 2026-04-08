#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${OUTPUT_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_cf.npy}"
: "${METHOD:=item2vec}"
: "${OUTPUT_DIM:=128}"
: "${WINDOW_SIZE:=5}"
: "${NEGATIVE_SAMPLES:=10}"
: "${EPOCHS:=5}"
: "${BATCH_SIZE:=4096}"
: "${LEARNING_RATE:=1e-3}"
: "${REG_WEIGHT:=1e-6}"
: "${SPPMI_SHIFT:=5.0}"
: "${DEVICE:=}"
: "${SEED:=0}"

export PYTHONUNBUFFERED=1

args=(
  scripts/generate_cf_item_embeddings.py
  --reviews-path "${REVIEWS_PATH}"
  --output-path "${OUTPUT_PATH}"
  --method "${METHOD}"
  --output-dim "${OUTPUT_DIM}"
  --window-size "${WINDOW_SIZE}"
  --negative-samples "${NEGATIVE_SAMPLES}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --reg-weight "${REG_WEIGHT}"
  --sppmi-shift "${SPPMI_SHIFT}"
  --seed "${SEED}"
  --overwrite
)

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

python "${args[@]}" "$@"
