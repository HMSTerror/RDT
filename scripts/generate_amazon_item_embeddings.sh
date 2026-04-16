#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${META_PATH:=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json}"
: "${OUTPUT_PATH:=data/Amazon_Music_And_Instruments/item_embeddings.npy}"
: "${TEXT_MODEL_NAME_OR_PATH:=google/t5-v1_1-small}"
: "${BATCH_SIZE:=64}"
: "${MAX_LENGTH:=96}"
: "${OUTPUT_DIM:=128}"
: "${DEVICE:=}"
: "${DTYPE:=auto}"
: "${SEED:=0}"
: "${LOCAL_FILES_ONLY:=0}"
: "${ITEM_UNIVERSE_SPLIT:=all}"
: "${SPLIT_MODE:=leave_last_two}"

export PYTHONUNBUFFERED=1

args=(
  scripts/generate_item_embeddings_from_meta.py
  --reviews-path "${REVIEWS_PATH}"
  --meta-path "${META_PATH}"
  --output-path "${OUTPUT_PATH}"
  --text-model-name-or-path "${TEXT_MODEL_NAME_OR_PATH}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --output-dim "${OUTPUT_DIM}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --item-universe-split "${ITEM_UNIVERSE_SPLIT}"
  --split-mode "${SPLIT_MODE}"
  --overwrite
)

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  args+=(--local-files-only)
fi

python "${args[@]}" "$@"
