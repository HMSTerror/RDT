# GenRec Non-DiT Pipeline

This document covers the pipeline stages that sit before the DiT generative
backbone:

1. Raw Data
2. Download + Preprocessing
3. Embedding Generation
4. Multimodal Fusion
5. Quantization

The goal is to make the repository look more like a UniGenRec-style toolbox,
where DiT is only one stage in a larger recommendation stack.

## Stage 1: Raw Data

The repository now has raw-data helpers under:
- [genrec/raw_data/amazon_music.py](/e:/RoboticsDiffusionTransformer/genrec/raw_data/amazon_music.py)

This stage is responsible for:
- locating reviews, metadata, images, and base embedding files
- standardizing raw input paths
- writing dataset manifests that later stages can consume

## Stage 2: Download + Preprocessing

Existing reusable assets:
- [scripts/download_amazon_music_dataset.sh](/e:/RoboticsDiffusionTransformer/scripts/download_amazon_music_dataset.sh)
- [preprocess_amazon.py](/e:/RoboticsDiffusionTransformer/preprocess_amazon.py)
- [scripts/preprocess_amazon_minimal.sh](/e:/RoboticsDiffusionTransformer/scripts/preprocess_amazon_minimal.sh)

New manifest helpers:
- [genrec/preprocessing/manifests.py](/e:/RoboticsDiffusionTransformer/genrec/preprocessing/manifests.py)

This stage should eventually produce:
- split-aware train/val/test buffers
- item metadata store
- item map / user map
- dense per-item representations for later quantization

## Stage 3: Embedding Generation

The repository already has one real branch implemented:
- text metadata embeddings:
  [scripts/generate_item_embeddings_from_meta.py](/e:/RoboticsDiffusionTransformer/scripts/generate_item_embeddings_from_meta.py)
- collaborative filtering embeddings:
  [scripts/generate_cf_item_embeddings.py](/e:/RoboticsDiffusionTransformer/scripts/generate_cf_item_embeddings.py)

The new embedding manifest layer is under:
- [genrec/embedding/manifest.py](/e:/RoboticsDiffusionTransformer/genrec/embedding/manifest.py)

Supported source names:
- `text`
- `image`
- `cf`
- `vlm`
- `fusion`

Currently implemented CF generators:
- `item_item_sppmi`
- `item2vec`
- `mf_bpr`

## Stage 4: Multimodal Fusion

Fusion helpers are implemented in:
- [genrec/fusion/strategies.py](/e:/RoboticsDiffusionTransformer/genrec/fusion/strategies.py)

Currently implemented strategies:
- `weighted_sum`
- `concat`
- `mean`

## Stage 5: Quantization

Quantization modules are implemented in:
- [genrec/quantization/pq.py](/e:/RoboticsDiffusionTransformer/genrec/quantization/pq.py)
- [genrec/quantization/opq.py](/e:/RoboticsDiffusionTransformer/genrec/quantization/opq.py)
- [genrec/quantization/rkmeans.py](/e:/RoboticsDiffusionTransformer/genrec/quantization/rkmeans.py)
- [genrec/quantization/rqvae.py](/e:/RoboticsDiffusionTransformer/genrec/quantization/rqvae.py)

Current support level:
- `PQ`: implemented
- `OPQ`: implemented as orthogonal rotation + PQ
- `RKMeans`: implemented as hierarchical k-means coding
- `RQ-VAE`: interface placeholder only

Semantic-ID build entrypoint:
- [scripts/build_semantic_ids.py](/e:/RoboticsDiffusionTransformer/scripts/build_semantic_ids.py)

Pipeline helper:
- [scripts/prepare_genrec_semantic_ids.sh](/e:/RoboticsDiffusionTransformer/scripts/prepare_genrec_semantic_ids.sh)

## Stage 6: Tokenized Samples

Tokenized sample builder:
- [scripts/build_tokenized_samples.py](/e:/RoboticsDiffusionTransformer/scripts/build_tokenized_samples.py)

Tokenized dataset loader:
- [genrec/data/tokenized_dataset.py](/e:/RoboticsDiffusionTransformer/genrec/data/tokenized_dataset.py)

Semantic-ID token layout helper:
- [genrec/tokenization/semantic_ids.py](/e:/RoboticsDiffusionTransformer/genrec/tokenization/semantic_ids.py)

Current sequence format:
- `[CLS]`
- history item semantic codes
- `[SEP]` after each valid history item
- `code_len` repeated `[MASK]` tokens for the target item
- `[EOS]`

Current optional condition branches returned by the dataset:
- `history_text_embeds`, `target_text_embed`, `pooled_text_embed`
- `history_cf_embeds`, `target_cf_embed`, `pooled_cf_embed`

This separation lets us improve embeddings, fusion, and quantization first,
then plug DiT in as the final generative backbone.
