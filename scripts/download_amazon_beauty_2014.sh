#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=data/Amazon_Beauty_2014}"
: "${REVIEWS_URL:=https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/reviews_Beauty.json.gz}"
: "${META_URL:=https://mcauleylab.ucsd.edu/public_datasets/data/amazon/categoryFiles/meta_Beauty.json.gz}"
: "${FORCE_DOWNLOAD:=0}"
: "${INSECURE_SKIP_TLS_VERIFY:=0}"
: "${CA_BUNDLE:=}"

mkdir -p "${DATA_ROOT}"

download_file() {
  local url="$1"
  local output="$2"

  if command -v aria2c >/dev/null 2>&1; then
    local aria2_args=(-x 16 -s 16 -k 1M --allow-overwrite=true -d "$(dirname "${output}")" -o "$(basename "${output}")")
    if [[ "${INSECURE_SKIP_TLS_VERIFY}" == "1" ]]; then
      echo "[warn] TLS certificate verification is disabled for aria2c."
      aria2_args+=(--check-certificate=false)
    fi
    aria2c "${aria2_args[@]}" "${url}"
    return
  fi

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

  echo "Neither aria2c, curl, nor wget is available. Please install one of them." >&2
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

REVIEWS_GZ="${DATA_ROOT}/reviews_Beauty.json.gz"
META_GZ="${DATA_ROOT}/meta_Beauty.json.gz"

ensure_file "reviews" "${REVIEWS_URL}" "${REVIEWS_GZ}"
ensure_file "metadata" "${META_URL}" "${META_GZ}"

cat <<EOF

Download complete.
Data root: ${DATA_ROOT}
Reviews : ${REVIEWS_GZ}
Meta    : ${META_GZ}

Suggested next step:
  bash scripts/prepare_amazon_beauty_2014.sh

Notes:
  - If your server lacks CA certificates, you can temporarily use:
      INSECURE_SKIP_TLS_VERIFY=1 bash scripts/download_amazon_beauty_2014.sh
  - A safer option is:
      CA_BUNDLE=/path/to/cacert.pem bash scripts/download_amazon_beauty_2014.sh
EOF
