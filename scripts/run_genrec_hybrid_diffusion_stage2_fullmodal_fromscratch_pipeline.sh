#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

: "${CONFIG_PATH:=configs/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v12_cf_every_other.yaml}"
: "${OUTPUT_DIR:=checkpoints/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v12_cf_every_other}"
: "${NUM_PROCESSES:=2}"
: "${MIXED_PRECISION:=bf16}"
: "${TRAIN_BATCH_SIZE:=8}"
: "${EVAL_BATCH_SIZE:=16}"
: "${NUM_TRAIN_EPOCHS:=30}"
: "${SAVE_EVERY_EPOCHS:=10}"
: "${EVAL_EVERY_EPOCHS:=10}"
: "${LOGGING_STEPS:=50}"
: "${TRAIN_LOG_PATH:=logs/genrec_stage2_fullmodal_fromscratch_train.log}"
: "${FINAL_EVAL_LOG_DIR:=logs/genrec_stage2_fullmodal_fromscratch_eval}"
: "${ABLATION_LOG_ROOT:=logs/genrec_stage2_fullmodal_fromscratch_ablation}"
: "${RUN_FINAL_FULL_EVAL:=0}"

mkdir -p "$(dirname "${TRAIN_LOG_PATH}")" "${FINAL_EVAL_LOG_DIR}" "${ABLATION_LOG_ROOT}"

echo "========== Stage2 Full-Modality From-Scratch Pipeline =========="
echo "config_path        : ${CONFIG_PATH}"
echo "output_dir         : ${OUTPUT_DIR}"
echo "num_train_epochs   : ${NUM_TRAIN_EPOCHS}"
echo "save_every_epochs  : ${SAVE_EVERY_EPOCHS}"
echo "eval_every_epochs  : ${EVAL_EVERY_EPOCHS}"
echo "train_log          : ${TRAIN_LOG_PATH}"
echo "run_final_full_eval: ${RUN_FINAL_FULL_EVAL}"

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  scripts/train_genrec_hybrid_diffusion.py \
  --config_path "${CONFIG_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --save_every_epochs "${SAVE_EVERY_EPOCHS}" \
  --eval_every_epochs "${EVAL_EVERY_EPOCHS}" \
  --logging_steps "${LOGGING_STEPS}" \
  --mixed_precision "${MIXED_PRECISION}" \
  2>&1 | tee "${TRAIN_LOG_PATH}"

FINAL_CHECKPOINT="${OUTPUT_DIR}/final"
if [[ ! -f "${FINAL_CHECKPOINT}/pytorch_model.bin" ]]; then
  latest_ckpt="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -1)"
  if [[ -n "${latest_ckpt}" && -f "${latest_ckpt}/pytorch_model.bin" ]]; then
    FINAL_CHECKPOINT="${latest_ckpt}"
  else
    echo "Could not resolve a final checkpoint under ${OUTPUT_DIR}."
    exit 1
  fi
fi

if [[ "${RUN_FINAL_FULL_EVAL}" == "1" ]]; then
  echo "========== Final Full Evaluation =========="
  CHECKPOINT="${FINAL_CHECKPOINT}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  CONFIG_PATH="${CONFIG_PATH}" \
  NUM_PROCESSES="${NUM_PROCESSES}" \
  MIXED_PRECISION="${MIXED_PRECISION}" \
  LOG_DIR="${FINAL_EVAL_LOG_DIR}" \
  bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_eval.sh
else
  echo "========== Skip Final Full Evaluation =========="
  echo "Ablation suite already includes a full run; set RUN_FINAL_FULL_EVAL=1 to enable a separate final full eval."
fi

echo "========== Final Ablation Suite =========="
CHECKPOINT="${FINAL_CHECKPOINT}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
CONFIG_PATH="${CONFIG_PATH}" \
NUM_PROCESSES="${NUM_PROCESSES}" \
MIXED_PRECISION="${MIXED_PRECISION}" \
ABLATION_LOG_ROOT="${ABLATION_LOG_ROOT}" \
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_ablation_suite.sh
