#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

: "${REVIEWS_PATH:=data/Amazon_Reviews_2023_All_Beauty/All_Beauty.jsonl}"
: "${META_PATH:=data/Amazon_Reviews_2023_All_Beauty/meta_All_Beauty.jsonl}"
: "${IMAGE_ROOT:=data/images/amazon_reviews23_all_beauty}"
: "${BUFFER_ROOT:=buffer/amazon_reviews23_all_beauty_real}"
: "${TEXT_EMBED_PATH:=data/Amazon_Reviews_2023_All_Beauty/item_embeddings_text.npy}"
: "${IMAGE_EMBED_PATH:=data/Amazon_Reviews_2023_All_Beauty/item_embeddings_image.npy}"
: "${CF_EMBED_PATH:=data/Amazon_Reviews_2023_All_Beauty/item_embeddings_cf.npy}"
: "${SEMANTIC_ID_ROOT:=data/Amazon_Reviews_2023_All_Beauty/semantic_ids_pq_cf_only}"
: "${TOKENIZED_ROOT:=data/Amazon_Reviews_2023_All_Beauty/tokenized_semantic_ids_cf_only}"

if [[ ! -f "${REVIEWS_PATH}" ]]; then
  echo "Review file not found: ${REVIEWS_PATH}" >&2
  echo "Place Amazon Reviews'23 All_Beauty reviews at the path above before running this script." >&2
  exit 1
fi

if [[ ! -f "${META_PATH}" ]]; then
  echo "Metadata file not found: ${META_PATH}" >&2
  echo "Place Amazon Reviews'23 All_Beauty metadata at the path above before running this script." >&2
  exit 1
fi

REVIEWS_PATH="${REVIEWS_PATH}" \
META_PATH="${META_PATH}" \
IMAGE_ROOT="${IMAGE_ROOT}" \
BUFFER_ROOT="${BUFFER_ROOT}" \
TEXT_EMBED_PATH="${TEXT_EMBED_PATH}" \
IMAGE_EMBED_PATH="${IMAGE_EMBED_PATH}" \
CF_EMBED_PATH="${CF_EMBED_PATH}" \
SEMANTIC_ID_ROOT="${SEMANTIC_ID_ROOT}" \
TOKENIZED_ROOT="${TOKENIZED_ROOT}" \
ENABLE_IMAGE=1 \
ENABLE_CF=1 \
ENABLE_FUSION=0 \
PRIMARY_EMBED_SOURCE=cf \
CF_METHOD="${CF_METHOD:-mf_bpr}" \
DOWNLOAD_MISSING_IMAGES="${DOWNLOAD_MISSING_IMAGES:-1}" \
VISION_MODEL_NAME_OR_PATH="${VISION_MODEL_NAME_OR_PATH:-google/siglip-base-patch16-224}" \
TEXT_MODEL_NAME_OR_PATH="${TEXT_MODEL_NAME_OR_PATH:-google/t5-v1_1-small}" \
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}" \
bash scripts/prepare_genrec_semantic_ids.sh

SOURCE_PATH="${TEXT_EMBED_PATH}" \
TARGET_PATH="${CF_EMBED_PATH}" \
OUTPUT_PATH="data/Amazon_Reviews_2023_All_Beauty/item_embeddings_text_to_cf.npy" \
bash scripts/generate_aligned_item_embeddings.sh

SOURCE_PATH="${IMAGE_EMBED_PATH}" \
TARGET_PATH="${CF_EMBED_PATH}" \
OUTPUT_PATH="data/Amazon_Reviews_2023_All_Beauty/item_embeddings_image_to_cf.npy" \
bash scripts/generate_aligned_item_embeddings.sh
