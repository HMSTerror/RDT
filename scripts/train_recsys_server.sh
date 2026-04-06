#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${TEXT_ENCODER_PATH:?Set TEXT_ENCODER_PATH to your local T5 checkpoint path}"
: "${VISION_ENCODER_PATH:?Set VISION_ENCODER_PATH to your local SigLIP checkpoint path}"
: "${BUFFER_ROOT:=/data/amazon_music_buffer}"
: "${IMAGE_ROOT:=/data/images}"
: "${OUTPUT_DIR:=checkpoints/recsys_amazon}"
: "${NUM_PROCESSES:=8}"
: "${MIXED_PRECISION:=bf16}"

export TOKENIZERS_PARALLELISM=false

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  main.py \
  --config_path configs/recsys_amazon.yaml \
  --buffer_root "${BUFFER_ROOT}" \
  --image_root "${IMAGE_ROOT}" \
  --pretrained_text_encoder_name_or_path "${TEXT_ENCODER_PATH}" \
  --pretrained_vision_encoder_name_or_path "${VISION_ENCODER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --train_batch_size 8 \
  --sample_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-6 \
  --sample_period 500 \
  --num_sample_batches 2 \
  --sample_topk 5,10,20 \
  --sample_exclude_history_items \
  "$@"
