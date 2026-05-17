#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${META_PATH:=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json}"
: "${EMBEDDING_PATH:=data/Amazon_Music_And_Instruments/item_embeddings.npy}"
: "${IMAGE_ROOT:=data/images}"
: "${OUTPUT_ROOT:=buffer/amazon_music}"
: "${HISTORY_LEN:=50}"
: "${CHUNK_SIZE:=1000}"
: "${DOWNLOAD_WORKERS:=16}"
: "${DOWNLOAD_TIMEOUT:=10}"
: "${SEED:=0}"
: "${SPLIT_MODE:=leave_last_two}"
: "${ITEM_UNIVERSE_SPLIT:=all}"
: "${DROP_LAST_INCOMPLETE_CHUNK:=0}"

export PYTHONUNBUFFERED=1

args=(
  preprocess_amazon.py
  --reviews-path "${REVIEWS_PATH}"
  --meta-path "${META_PATH}"
  --embedding-path "${EMBEDDING_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --history-len "${HISTORY_LEN}"
  --chunk-size "${CHUNK_SIZE}"
  --download-workers "${DOWNLOAD_WORKERS}"
  --download-timeout "${DOWNLOAD_TIMEOUT}"
  --seed "${SEED}"
  --split-mode "${SPLIT_MODE}"
  --item-universe-split "${ITEM_UNIVERSE_SPLIT}"
  --overwrite
)

if [[ "${DROP_LAST_INCOMPLETE_CHUNK}" == "1" ]]; then
  args+=(--drop-last-incomplete-chunk)
fi

python "${args[@]}" "$@"
