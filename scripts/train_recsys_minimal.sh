#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export TOKENIZERS_PARALLELISM=false

python main.py \
  --config_path configs/recsys_amazon_smoke.yaml \
  --pretrained_text_encoder_name_or_path dummy \
  --pretrained_vision_encoder_name_or_path dummy \
  --output_dir checkpoints/recsys_smoke \
  --train_batch_size 2 \
  --sample_batch_size 2 \
  --num_train_epochs 1 \
  --max_train_steps 2 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 1 \
  --dataloader_num_workers 0 \
  --sample_period 1 \
  --num_sample_batches 1 \
  --sample_topk 5,10 \
  --sample_exclude_history_items \
  --report_to none \
  "$@"
