#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${META_PATH:=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json}"
: "${IMAGE_ROOT:=data/images}"
: "${OUTPUT_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_image.npy}"
: "${VISION_MODEL_NAME_OR_PATH:=google/siglip-base-patch16-224}"
: "${BATCH_SIZE:=64}"
: "${OUTPUT_DIM:=128}"
: "${DEVICE:=}"
: "${DTYPE:=auto}"
: "${SEED:=0}"
: "${LOCAL_FILES_ONLY:=0}"
: "${DOWNLOAD_MISSING:=0}"
: "${DOWNLOAD_WORKERS:=16}"
: "${DOWNLOAD_TIMEOUT:=10}"
: "${ALLOW_ALL_MISSING_IMAGES:=0}"
: "${ITEM_UNIVERSE_SPLIT:=all}"
: "${SPLIT_MODE:=leave_last_two}"

export PYTHONUNBUFFERED=1

args=(
  scripts/generate_image_embeddings_from_images.py
  --reviews-path "${REVIEWS_PATH}"
  --meta-path "${META_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --vision-model-name-or-path "${VISION_MODEL_NAME_OR_PATH}"
  --batch-size "${BATCH_SIZE}"
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

if [[ "${DOWNLOAD_MISSING}" == "1" ]]; then
  args+=(--download-missing --download-workers "${DOWNLOAD_WORKERS}" --download-timeout "${DOWNLOAD_TIMEOUT}")
fi

if [[ "${ALLOW_ALL_MISSING_IMAGES}" == "1" ]]; then
  args+=(--allow-all-missing)
fi

python "${args[@]}" "$@"
