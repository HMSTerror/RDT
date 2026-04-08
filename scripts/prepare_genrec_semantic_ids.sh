#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

: "${REVIEWS_PATH:=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json}"
: "${META_PATH:=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json}"
: "${IMAGE_ROOT:=data/images}"
: "${BUFFER_ROOT:=buffer/amazon_music_real}"
: "${TEXT_EMBED_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_text.npy}"
: "${IMAGE_EMBED_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_image.npy}"
: "${CF_EMBED_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_cf.npy}"
: "${FUSED_EMBED_PATH:=data/Amazon_Music_And_Instruments/item_embeddings_fused.npy}"
: "${SEMANTIC_ID_ROOT:=data/Amazon_Music_And_Instruments/semantic_ids_pq}"
: "${TOKENIZED_ROOT:=data/Amazon_Music_And_Instruments/tokenized_semantic_ids}"
: "${ITEM_MAP_PATH:=${BUFFER_ROOT}/train/item_map.json}"
: "${QUANT_METHOD:=pq}"

echo "========== [1/7] text embeddings =========="
OUTPUT_PATH="${TEXT_EMBED_PATH}" \
REVIEWS_PATH="${REVIEWS_PATH}" \
META_PATH="${META_PATH}" \
bash scripts/generate_amazon_item_embeddings.sh

if [[ "${ENABLE_IMAGE:-1}" == "1" ]]; then
  echo "========== [2/7] image embeddings =========="
  OUTPUT_PATH="${IMAGE_EMBED_PATH}" \
  REVIEWS_PATH="${REVIEWS_PATH}" \
  META_PATH="${META_PATH}" \
  IMAGE_ROOT="${IMAGE_ROOT}" \
  VISION_MODEL_NAME_OR_PATH="${VISION_MODEL_NAME_OR_PATH:-google/siglip-base-patch16-224}" \
  DEVICE="${IMAGE_DEVICE:-${DEVICE:-}}" \
  DTYPE="${IMAGE_DTYPE:-${DTYPE:-auto}}" \
  LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}" \
  DOWNLOAD_MISSING="${DOWNLOAD_MISSING_IMAGES:-0}" \
  DOWNLOAD_WORKERS="${IMAGE_DOWNLOAD_WORKERS:-16}" \
  DOWNLOAD_TIMEOUT="${IMAGE_DOWNLOAD_TIMEOUT:-10}" \
  bash scripts/generate_amazon_image_embeddings.sh
else
  echo "========== [2/7] image embeddings skipped =========="
fi

if [[ "${ENABLE_CF:-0}" == "1" ]]; then
  echo "========== [3/7] CF embeddings =========="
  OUTPUT_PATH="${CF_EMBED_PATH}" \
  REVIEWS_PATH="${REVIEWS_PATH}" \
  METHOD="${CF_METHOD:-item2vec}" \
  WINDOW_SIZE="${CF_WINDOW_SIZE:-5}" \
  NEGATIVE_SAMPLES="${CF_NEGATIVE_SAMPLES:-10}" \
  EPOCHS="${CF_EPOCHS:-5}" \
  BATCH_SIZE="${CF_BATCH_SIZE:-4096}" \
  LEARNING_RATE="${CF_LEARNING_RATE:-1e-3}" \
  REG_WEIGHT="${CF_REG_WEIGHT:-1e-6}" \
  DEVICE="${CF_DEVICE:-${DEVICE:-}}" \
  bash scripts/generate_amazon_cf_item_embeddings.sh
else
  echo "========== [3/7] CF embeddings skipped =========="
fi

echo "========== [4/7] preprocess buffer =========="
REVIEWS_PATH="${REVIEWS_PATH}" \
META_PATH="${META_PATH}" \
EMBEDDING_PATH="${TEXT_EMBED_PATH}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
OUTPUT_ROOT="${BUFFER_ROOT}" \
SPLIT_MODE=leave_last_two \
bash scripts/preprocess_amazon_minimal.sh

INPUT_EMBED_PATH="${TEXT_EMBED_PATH}"
if [[ "${ENABLE_FUSION:-0}" == "1" ]]; then
  echo "========== [5/7] multimodal fusion =========="
  fusion_args=(
    scripts/build_multimodal_item_embeddings.py
    --text-path "${TEXT_EMBED_PATH}"
    --output-path "${FUSED_EMBED_PATH}"
    --strategy "${FUSION_STRATEGY:-weighted_sum}"
    --normalize-each
    --normalize-output
  )
  if [[ "${ENABLE_IMAGE:-1}" == "1" ]]; then
    fusion_args+=(--image-path "${IMAGE_EMBED_PATH}" --image-weight "${FUSION_IMAGE_WEIGHT:-1.0}")
  fi
  if [[ "${ENABLE_CF:-0}" == "1" ]]; then
    fusion_args+=(--cf-path "${CF_EMBED_PATH}" --cf-weight "${FUSION_CF_WEIGHT:-1.0}")
  fi
  fusion_args+=(--text-weight "${FUSION_TEXT_WEIGHT:-1.0}")
  python "${fusion_args[@]}"
  INPUT_EMBED_PATH="${FUSED_EMBED_PATH}"
else
  echo "========== [5/7] multimodal fusion skipped =========="
fi

echo "========== [6/7] semantic ID quantization =========="
python scripts/build_semantic_ids.py \
  --embeddings-path "${INPUT_EMBED_PATH}" \
  --item-map-path "${ITEM_MAP_PATH}" \
  --output-root "${SEMANTIC_ID_ROOT}" \
  --method "${QUANT_METHOD}" \
  --num-subspaces "${NUM_SUBSPACES:-4}" \
  --codebook-size "${CODEBOOK_SIZE:-256}" \
  --levels "${RKMEANS_LEVELS:-4}" \
  --branching-factor "${RKMEANS_BRANCHING_FACTOR:-16}" \
  --max-iter "${MAX_ITER:-25}" \
  --seed "${SEED:-0}"

echo "========== [7/7] tokenized samples =========="
python scripts/build_tokenized_samples.py \
  --buffer-root "${BUFFER_ROOT}" \
  --semantic-id-root "${SEMANTIC_ID_ROOT}" \
  --output-root "${TOKENIZED_ROOT}" \
  --splits "${TOKENIZED_SPLITS:-train,val,test}" \
  --pad-token-id "${PAD_TOKEN_ID:-0}" \
  --mask-token-id "${MASK_TOKEN_ID:-1}" \
  --cls-token-id "${CLS_TOKEN_ID:-2}" \
  --sep-token-id "${SEP_TOKEN_ID:-3}" \
  --eos-token-id "${EOS_TOKEN_ID:-4}"

echo "========== pipeline finished =========="
