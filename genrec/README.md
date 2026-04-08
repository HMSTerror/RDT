# GenRec Workspace

This package is the new development area for the GenRec-style pipeline.

Design intent:
- keep the current continuous-latent RecSys-DiT code in the repository
- preserve a legacy snapshot under `old/legacy_recsys_dit/`
- build the new semantic-ID + generative-decoding workflow in parallel

Planned submodules:
- `genrec/raw_data/`
  raw dataset discovery and dataset manifests
- `genrec/preprocessing/`
  preprocessing-stage manifests and pipeline wiring
- `genrec/embedding/`
  dense embedding artifacts for text / image / CF / VLM branches
- `genrec/fusion/`
  multimodal fusion strategies over dense item embeddings
- `genrec/tokenization/`
  build semantic IDs from dense item embeddings
- `genrec/quantization/`
  classical quantizers and RQ-VAE placeholder interfaces
- `genrec/data/`
  tokenized datasets and collators
- `genrec/models/`
  DiT-based generative backbone and heads
- `genrec/inference/`
  greedy / beam / prefix-tree decode

The high-level migration plan is documented in:
- [docs/genrec_dit_migration_design.md](/e:/RoboticsDiffusionTransformer/docs/genrec_dit_migration_design.md)

Implemented in this stage:
- raw-data manifest helpers for Amazon Music
- pipeline manifests for preprocessing / embedding / fusion / quantization
- collaborative item embedding generation:
  - item-item SPPMI
  - item2vec
  - MF-BPR
- weighted-sum and concat fusion helpers
- classical semantic-ID quantizers:
  - PQ
  - OPQ
  - RKMeans
- RQ-VAE interface placeholder
- tokenized semantic-ID sample builder and dataset loader
- multi-branch condition loading for:
  - text history / pooled summary
  - CF history / pooled summary
- DiT-style semantic-token backbone:
  [genrec/models/genrec_dit.py](/e:/RoboticsDiffusionTransformer/genrec/models/genrec_dit.py)
- prefix-constrained semantic-ID decoding helpers:
  [genrec/inference/semantic_decoder.py](/e:/RoboticsDiffusionTransformer/genrec/inference/semantic_decoder.py)
- standalone GenRec training entry:
  [scripts/train_genrec_dit.py](/e:/RoboticsDiffusionTransformer/scripts/train_genrec_dit.py)
- standalone grouped evaluation entry:
  [scripts/eval_genrec_dit.py](/e:/RoboticsDiffusionTransformer/scripts/eval_genrec_dit.py)
- end-to-end pipeline wrapper:
  [scripts/run_genrec_dit_pipeline.sh](/e:/RoboticsDiffusionTransformer/scripts/run_genrec_dit_pipeline.sh)
- CLI semantic-ID builder:
  [scripts/build_semantic_ids.py](/e:/RoboticsDiffusionTransformer/scripts/build_semantic_ids.py)

Quick start:

```bash
python scripts/train_genrec_dit.py \
  --config_path configs/genrec_dit_amazon.yaml \
  --train_batch_size 8 \
  --mixed_precision bf16
```

Grouped test evaluation:

```bash
python scripts/eval_genrec_dit.py \
  --config_path configs/genrec_dit_amazon.yaml \
  --checkpoint checkpoints/genrec_dit_amazon/final \
  --split test \
  --batch_size 16 \
  --topk 5,10,20 \
  --exclude_history_items \
  --group_strategy equal_items \
  --frequency_source_split train
```
