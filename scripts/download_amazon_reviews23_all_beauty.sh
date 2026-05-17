#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=data/Amazon_Reviews_2023_All_Beauty}"
: "${REVIEWS_URL:=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl}"
: "${META_URL:=https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl}"
: "${FORCE_DOWNLOAD:=0}"
: "${INSECURE_SKIP_TLS_VERIFY:=0}"
: "${CA_BUNDLE:=}"

mkdir -p "${DATA_ROOT}"

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    local curl_args=(-L --fail --retry 3 --retry-delay 2 -o "${output}")
    if [[ -n "${CA_BUNDLE}" ]]; then
      curl_args+=(--cacert "${CA_BUNDLE}")
    fi
    if [[ "${INSECURE_SKIP_TLS_VERIFY}" == "1" ]]; then
      echo "[warn] TLS certificate verification is disabled for curl."
      curl_args+=(-k)
    fi
    curl "${curl_args[@]}" "${url}"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    local wget_args=(-O "${output}")
    if [[ -n "${CA_BUNDLE}" ]]; then
      wget_args+=(--ca-certificate="${CA_BUNDLE}")
    fi
    if [[ "${INSECURE_SKIP_TLS_VERIFY}" == "1" ]]; then
      echo "[warn] TLS certificate verification is disabled for wget."
      wget_args+=(--no-check-certificate)
    fi
    wget "${wget_args[@]}" "${url}"
    return
  fi

  echo "Neither curl nor wget is available. Please install one of them." >&2
  exit 1
}

ensure_file() {
  local label="$1"
  local url="$2"
  local output="$3"

  if [[ -s "${output}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
    echo "[skip] ${label} already exists: ${output}"
    return
  fi

  echo "[download] ${label}"
  download_file "${url}" "${output}"
}

REVIEWS_JSONL="${DATA_ROOT}/All_Beauty.jsonl"
META_JSONL="${DATA_ROOT}/meta_All_Beauty.jsonl"

ensure_file "reviews" "${REVIEWS_URL}" "${REVIEWS_JSONL}"
ensure_file "metadata" "${META_URL}" "${META_JSONL}"

cat <<EOF

Download complete.
Data root: ${DATA_ROOT}
Reviews : ${REVIEWS_JSONL}
Meta    : ${META_JSONL}

Optional next step:
  bash scripts/prepare_amazon_reviews23_all_beauty.sh

Notes:
  - If your server lacks CA certificates, you can temporarily use:
      INSECURE_SKIP_TLS_VERIFY=1 bash scripts/download_amazon_reviews23_all_beauty.sh
  - A safer option is:
      CA_BUNDLE=/path/to/cacert.pem bash scripts/download_amazon_reviews23_all_beauty.sh
EOF
