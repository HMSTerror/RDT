#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${IMAGE_ROOT:=data/images}"
: "${OUTPUT_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_image.npy}"
: "${VISION_MODEL_NAME_OR_PATH:=google/siglip-base-patch16-224}"
: "${BATCH_SIZE:=64}"
: "${OUTPUT_DIM:=128}"
: "${DEVICE:=}"
: "${DTYPE:=auto}"
: "${SEED:=0}"
: "${LOCAL_FILES_ONLY:=0}"

export PYTHONUNBUFFERED=1

args=(
  scripts/generate_image_embeddings_from_images.py
  --reviews-path "${REVIEWS_PATH}"
  --image-root "${IMAGE_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --vision-model-name-or-path "${VISION_MODEL_NAME_OR_PATH}"
  --batch-size "${BATCH_SIZE}"
  --output-dim "${OUTPUT_DIM}"
  --dtype "${DTYPE}"
  --seed "${SEED}"
  --overwrite
)

if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  args+=(--local-files-only)
fi

python "${args[@]}" "$@"
