#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=data/Amazon_Music_And_Instruments}"
: "${REVIEWS_URL:=https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz}"
: "${META_URL:=https://jmcauley.ucsd.edu/pml_data/meta_Musical_Instruments.json.gz}"
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

extract_gzip_to_json() {
  local gzip_path="$1"
  local json_path="$2"

  python -c "import gzip, shutil; from pathlib import Path; src=Path(r'''${gzip_path}'''); dst=Path(r'''${json_path}'''); dst.parent.mkdir(parents=True, exist_ok=True); \
with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout: shutil.copyfileobj(fin, fout)"
}

ensure_json_file() {
  local label="$1"
  local url="$2"
  local gzip_path="$3"
  local json_path="$4"

  if [[ -s "${json_path}" && "${FORCE_DOWNLOAD}" != "1" ]]; then
    echo "[skip] ${label} already exists: ${json_path}"
    return
  fi

  if [[ ! -s "${gzip_path}" || "${FORCE_DOWNLOAD}" == "1" ]]; then
    echo "[download] ${label}"
    download_file "${url}" "${gzip_path}"
  else
    echo "[reuse] ${label} archive already exists: ${gzip_path}"
  fi

  echo "[extract] ${label} -> ${json_path}"
  extract_gzip_to_json "${gzip_path}" "${json_path}"
}

REVIEWS_GZ="${DATA_ROOT}/Musical_Instruments_5.json.gz"
REVIEWS_JSON="${DATA_ROOT}/Musical_Instruments_5.json"
META_GZ="${DATA_ROOT}/meta_Musical_Instruments.json.gz"
META_JSON="${DATA_ROOT}/meta_Musical_Instruments.json"

ensure_json_file "reviews" "${REVIEWS_URL}" "${REVIEWS_GZ}" "${REVIEWS_JSON}"
ensure_json_file "metadata" "${META_URL}" "${META_GZ}" "${META_JSON}"

cat <<EOF

Download complete.
Data root: ${DATA_ROOT}
Reviews : ${REVIEWS_JSON}
Meta    : ${META_JSON}

Optional next step:
  TEXT_MODEL_NAME_OR_PATH=/path/to/local_t5_or_hf_id bash scripts/generate_amazon_item_embeddings.sh
  bash scripts/preprocess_amazon_minimal.sh

Notes:
  - item_embeddings.npy is not downloaded by this script.
  - You can generate metadata-based semantic embeddings with:
      bash scripts/generate_amazon_item_embeddings.sh
  - preprocess_amazon.py will generate dummy 128-d item embeddings only if that file is absent.
  - If your server lacks CA certificates, you can temporarily use:
      INSECURE_SKIP_TLS_VERIFY=1 bash scripts/download_amazon_music_dataset.sh
  - A safer option is:
      CA_BUNDLE=/path/to/cacert.pem bash scripts/download_amazon_music_dataset.sh
EOF
