# RecSys-DiT Server Workflow

This document describes the recommended end-to-end server workflow:

1. download the raw Amazon Musical Instruments dataset into `data/`
2. generate real metadata-based `item_embeddings.npy`
3. preprocess the raw files into a lightweight buffer
4. train RecSys-DiT
5. run offline top-k retrieval evaluation

## 1. Directory Layout

Suggested server layout:

```bash
/workspace/RDT
/workspace/RDT/data
/workspace/RDT/data/images
/workspace/RDT/data/Amazon_Music_And_Instruments
/data/amazon_music_buffer
/models/t5-v1_1-xxl
/models/siglip-base-patch16-224
/exp/recsys_dit
```

## 2. Install Dependencies

Install a matching `torch` / `torchvision` build for your CUDA version first, then:

```bash
pip install -r requirements.txt
pip install -r requirements_data.txt
```

## 3. Download Raw Amazon Data

The raw Amazon dataset is not tracked by Git. Download it into `data/Amazon_Music_And_Instruments` with:

```bash
cd /workspace/RDT
bash scripts/download_amazon_music_dataset.sh
```

This script downloads and extracts:

- `data/Amazon_Music_And_Instruments/Musical_Instruments_5.json`
- `data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json`

If you want to force a re-download:

```bash
FORCE_DOWNLOAD=1 bash scripts/download_amazon_music_dataset.sh
```

If you want to place the raw files somewhere else under `data/`:

```bash
DATA_ROOT=data/Amazon_Music_And_Instruments \
bash scripts/download_amazon_music_dataset.sh
```

Important note:

- `item_embeddings.npy` is not downloaded by this script.
- If that file is absent, `preprocess_amazon.py` will generate dummy item embeddings for pipeline testing.

## 4. Generate Real Item Embeddings

Use a local or downloadable text encoder to turn item metadata into semantic 128-d vectors.
If you already downloaded `t5-v1_1-small` locally, that is a good lightweight default:

```bash
cd /workspace/RDT

TEXT_MODEL_NAME_OR_PATH=/models/t5-v1_1-small \
LOCAL_FILES_ONLY=1 \
bash scripts/generate_amazon_item_embeddings.sh
```

This writes:

- `data/Amazon_Music_And_Instruments/item_embeddings.npy`
- `data/Amazon_Music_And_Instruments/item_embeddings.meta.json`

Notes:

- The script uses the same 5-core filtered item ordering as `preprocess_amazon.py`.
- The output is metadata-based, not random, so retrieval metrics become meaningful.
- This still is not the same thing as a collaborative item2vec embedding learned from user behavior.

## 5. Optional Local Images

If you already have product images, place them under:

```bash
data/images/{asin}.jpg
```

If images are missing, preprocessing can optionally try to download them from metadata URLs unless you pass `--skip-download`.

## 6. Preprocess Into a Lightweight Buffer

The most convenient wrapper is:

```bash
REVIEWS_PATH=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json \
META_PATH=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json \
EMBEDDING_PATH=data/Amazon_Music_And_Instruments/item_embeddings.npy \
IMAGE_ROOT=data/images \
OUTPUT_ROOT=/data/amazon_music_buffer \
bash scripts/preprocess_amazon_minimal.sh
```

For a smaller debug build:

```bash
REVIEWS_PATH=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json \
META_PATH=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json \
EMBEDDING_PATH=data/Amazon_Music_And_Instruments/item_embeddings.npy \
IMAGE_ROOT=data/images \
OUTPUT_ROOT=/data/amazon_music_buffer_debug \
bash scripts/preprocess_amazon_minimal.sh --max-samples 10000 --skip-download
```

## 7. Train on the Server

Use the server training wrapper:

```bash
TEXT_ENCODER_PATH=/models/t5-v1_1-xxl \
VISION_ENCODER_PATH=/models/siglip-base-patch16-224 \
BUFFER_ROOT=/data/amazon_music_buffer \
IMAGE_ROOT=data/images \
OUTPUT_DIR=/exp/recsys_dit/run_001 \
NUM_PROCESSES=8 \
MIXED_PRECISION=bf16 \
bash scripts/train_recsys_server.sh
```

You can append custom arguments, for example:

```bash
TEXT_ENCODER_PATH=/models/t5-v1_1-xxl \
VISION_ENCODER_PATH=/models/siglip-base-patch16-224 \
BUFFER_ROOT=/data/amazon_music_buffer \
IMAGE_ROOT=data/images \
OUTPUT_DIR=/exp/recsys_dit/run_002 \
bash scripts/train_recsys_server.sh \
  --train_batch_size 4 \
  --sample_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3e-5 \
  --sample_period 200 \
  --num_sample_batches 4 \
  --sample_topk 5,10,20,50
```

During training, sampling can automatically report:

- `NDCG@K`
- `Hit@K`
- `MRR@K`

## 8. Offline Retrieval Evaluation

After training, evaluate a checkpoint with:

```bash
python retrieve_topk.py \
  --config_path configs/recsys_amazon.yaml \
  --checkpoint /exp/recsys_dit/run_001 \
  --buffer_root /data/amazon_music_buffer \
  --image_root data/images \
  --pretrained_text_encoder_name_or_path /models/t5-v1_1-xxl \
  --pretrained_vision_encoder_name_or_path /models/siglip-base-patch16-224 \
  --batch_size 8 \
  --topk 5,10,20,50 \
  --similarity cosine \
  --exclude_history_items
```

To save per-sample retrieval results:

```bash
python retrieve_topk.py \
  --config_path configs/recsys_amazon.yaml \
  --checkpoint /exp/recsys_dit/run_001 \
  --buffer_root /data/amazon_music_buffer \
  --image_root data/images \
  --pretrained_text_encoder_name_or_path /models/t5-v1_1-xxl \
  --pretrained_vision_encoder_name_or_path /models/siglip-base-patch16-224 \
  --batch_size 8 \
  --topk 5,10,20,50 \
  --similarity cosine \
  --exclude_history_items \
  --save_jsonl /exp/recsys_dit/run_001/retrieve_topk.jsonl
```

## 9. Full Copy-Paste Workflow

```bash
cd /workspace/RDT

bash scripts/download_amazon_music_dataset.sh

TEXT_MODEL_NAME_OR_PATH=/models/t5-v1_1-small \
LOCAL_FILES_ONLY=1 \
bash scripts/generate_amazon_item_embeddings.sh

REVIEWS_PATH=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json \
META_PATH=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json \
EMBEDDING_PATH=data/Amazon_Music_And_Instruments/item_embeddings.npy \
IMAGE_ROOT=data/images \
OUTPUT_ROOT=/data/amazon_music_buffer \
bash scripts/preprocess_amazon_minimal.sh

TEXT_ENCODER_PATH=/models/t5-v1_1-xxl \
VISION_ENCODER_PATH=/models/siglip-base-patch16-224 \
BUFFER_ROOT=/data/amazon_music_buffer \
IMAGE_ROOT=data/images \
OUTPUT_DIR=/exp/recsys_dit/run_001 \
NUM_PROCESSES=8 \
MIXED_PRECISION=bf16 \
bash scripts/train_recsys_server.sh

python retrieve_topk.py \
  --config_path configs/recsys_amazon.yaml \
  --checkpoint /exp/recsys_dit/run_001 \
  --buffer_root /data/amazon_music_buffer \
  --image_root data/images \
  --pretrained_text_encoder_name_or_path /models/t5-v1_1-xxl \
  --pretrained_vision_encoder_name_or_path /models/siglip-base-patch16-224 \
  --batch_size 8 \
  --topk 5,10,20,50 \
  --similarity cosine \
  --exclude_history_items \
  --save_jsonl /exp/recsys_dit/run_001/retrieve_topk.jsonl
```

## 10. Minimal Smoke Workflow

If you only want to validate the environment first:

```bash
bash scripts/download_amazon_music_dataset.sh

REVIEWS_PATH=data/Amazon_Music_And_Instruments/Musical_Instruments_5.json \
META_PATH=data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json \
EMBEDDING_PATH=data/Amazon_Music_And_Instruments/item_embeddings.npy \
IMAGE_ROOT=data/images \
OUTPUT_ROOT=/data/amazon_music_buffer_debug \
bash scripts/preprocess_amazon_minimal.sh --max-samples 1000 --skip-download

python main.py \
  --config_path configs/recsys_amazon_smoke.yaml \
  --buffer_root /data/amazon_music_buffer_debug \
  --image_root data/images \
  --pretrained_text_encoder_name_or_path dummy \
  --pretrained_vision_encoder_name_or_path dummy \
  --output_dir /exp/recsys_dit/smoke \
  --train_batch_size 2 \
  --sample_batch_size 2 \
  --max_train_steps 2 \
  --sample_period 1 \
  --num_sample_batches 1 \
  --sample_topk 5,10 \
  --report_to none \
  --mixed_precision no
```

## 11. Notes

- The raw dataset directory is intentionally ignored by Git:
  `data/Amazon_Music_And_Instruments/`
- If product images are missing and the server has no internet access, add `--skip-download` during preprocessing.
- If `item_embeddings.npy` is absent, preprocessing falls back to dummy vectors intended only for smoke testing.
- If GPU memory is tight, reduce `train_batch_size` first and then increase `gradient_accumulation_steps`.
