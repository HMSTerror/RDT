#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${DATA_ROOT:=data/Amazon_Music_And_Instruments}"
: "${REVIEWS_URL:=https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz}"
: "${META_URL:=https://jmcauley.ucsd.edu/pml_data/meta_Musical_Instruments.json.gz}"
: "${FORCE_DOWNLOAD:=0}"

mkdir -p "${DATA_ROOT}"

download_file() {
  local url="$1"
  local output="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 --retry-delay 2 -o "${output}" "${url}"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
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
  bash scripts/preprocess_amazon_minimal.sh

Notes:
  - item_embeddings.npy is not downloaded by this script.
  - preprocess_amazon.py will generate dummy 128-d item embeddings if that file is absent.
EOF
