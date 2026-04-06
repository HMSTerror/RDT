# RecSys-DiT

This repository is now a recommendation-only diffusion project. The original robotics data pipelines, evaluation tools, and TensorFlow preprocessing stack have been removed from the active path.

The active scope is:

- multimodal sequence recommendation
- diffusion over continuous 128-d item embeddings
- Amazon-style offline preprocessing into lightweight chunk buffers
- training-time sampling with `NDCG@K`, `Hit@K`, and `MRR@K`

## Recsys-Only Layout

```text
configs/
  recsys_amazon.yaml
  recsys_amazon_smoke.yaml
models/
  multimodal_encoder/
  rdt/
  rdt_runner.py
scripts/
  download_amazon_music_dataset.sh
  train_recsys_minimal.sh
  train_recsys_server.sh
train/
  dataset.py
  sample.py
  train.py
main.py
preprocess_amazon.py
retrieve_topk.py
test_overfit.py
```

## Install

`requirements.txt` is now runtime-only for RecSys-DiT, and `requirements_data.txt` is preprocessing-only.

```bash
# install torch / torchvision separately for your CUDA version first
pip install -r requirements.txt
pip install -r requirements_data.txt
```

## Download Raw Data

The raw Amazon Musical Instruments files are intentionally not tracked by Git.

Download them into `data/Amazon_Music_And_Instruments` with:

```bash
bash scripts/download_amazon_music_dataset.sh
```

This script creates:

- `data/Amazon_Music_And_Instruments/Musical_Instruments_5.json`
- `data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json`

If you want to override the target folder:

```bash
DATA_ROOT=data/Amazon_Music_And_Instruments \
bash scripts/download_amazon_music_dataset.sh
```

## Current Pipeline

1. `scripts/download_amazon_music_dataset.sh`
2. `preprocess_amazon.py`
3. `train/dataset.py`
4. `models/multimodal_encoder/condition_encoder.py`
5. `models/rdt/model.py`
6. `models/rdt/blocks.py`
7. `models/rdt_runner.py`
8. `train/train.py`
9. `train/sample.py`
10. `retrieve_topk.py`

## Buffer Format

The preferred training input is the lightweight recommendation buffer produced by `preprocess_amazon.py`.

The buffer contains:

- `item_embeddings.npy`
- `item_meta.json`
- `item_map.json`
- `user_map.json`
- `stats.json`
- `chunk_k/samples.npz`
- `chunk_k/dirty_bit`

Each stored sample keeps only lightweight IDs and masks. Image tensors and item embeddings are reconstructed dynamically at training time from the global item store.

## Minimal Smoke Run

The quickest fully offline path uses dummy encoders plus the local `.tmp_amazon_light` buffer:

```bash
bash scripts/train_recsys_minimal.sh
```

Equivalent direct command:

```bash
python main.py \
  --config_path configs/recsys_amazon_smoke.yaml \
  --pretrained_text_encoder_name_or_path dummy \
  --pretrained_vision_encoder_name_or_path dummy \
  --output_dir checkpoints/recsys_smoke \
  --train_batch_size 2 \
  --sample_batch_size 2 \
  --max_train_steps 2 \
  --sample_period 1 \
  --num_sample_batches 1 \
  --sample_topk 5,10 \
  --sample_exclude_history_items \
  --report_to none
```

## Server Training

完整的服务器闭环文档见：

- [Recsys Server Workflow](docs/recsys_server_workflow.md)

For real training on a Linux server, keep the large encoders local on disk and pass dataset paths at launch time instead of editing YAML by hand.

```bash
TEXT_ENCODER_PATH=/models/t5-v1_1-xxl \
VISION_ENCODER_PATH=/models/siglip-base-patch16-224 \
BUFFER_ROOT=/data/amazon_music_buffer \
IMAGE_ROOT=/data/images \
OUTPUT_DIR=/exp/recsys_dit \
bash scripts/train_recsys_server.sh
```

Equivalent direct launch command:

```bash
accelerate launch \
  --num_processes 8 \
  --mixed_precision bf16 \
  main.py \
  --config_path configs/recsys_amazon.yaml \
  --buffer_root /data/amazon_music_buffer \
  --image_root /data/images \
  --pretrained_text_encoder_name_or_path /models/t5-v1_1-xxl \
  --pretrained_vision_encoder_name_or_path /models/siglip-base-patch16-224 \
  --output_dir /exp/recsys_dit \
  --train_batch_size 8 \
  --sample_batch_size 8 \
  --sample_period 500 \
  --num_sample_batches 2 \
  --sample_topk 5,10,20 \
  --sample_exclude_history_items
```

Important runtime notes:

- `action_dim` is enforced to `128`
- `action_chunk_size` is enforced to `1`
- inference sampling never consumes the real target image
- `main.py` now supports `--buffer_root` and `--image_root` overrides
- `main.py` now supports `--report_to none` for smoke tests
- `main.py` now supports `dummy` text and vision encoders for offline validation

## Retrieval Evaluation

Run offline retrieval evaluation with:

```bash
python retrieve_topk.py \
  --config_path configs/recsys_amazon.yaml \
  --checkpoint /path/to/checkpoint \
  --pretrained_text_encoder_name_or_path /path/to/t5 \
  --pretrained_vision_encoder_name_or_path /path/to/siglip \
  --topk 5,10,20
```

This reports:

- `mean_rank`
- `hit@k`
- `recall@k`
- `mrr@k`
- `ndcg@k`

## Gradient Smoke Test

Use the fixed dummy-batch overfit script to verify that gradients flow end to end:

```bash
python test_overfit.py --steps 50 --batch-size 2 --history-len 4 --lr 1e-4
```
