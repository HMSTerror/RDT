# GenRec-Style DiT Migration Design

## Goal

Refactor this repository from a continuous embedding regression pipeline into a
GenRec-style generative recommendation pipeline, while reusing the existing DiT
components as the new generative backbone.

The target design is inspired by:
- UniGenRec repository:
  https://github.com/hupeiyu21/UniGenRec-A-universal-generative-recommendation-toolbox
- UniGenRec pipeline description:
  Representation -> Tokenization -> Modeling -> Training -> Inference
- UniGenRec module split:
  `preprocessing/`, `quantization/`, `recommendation/`, `evaluation/`

## Why This Migration

The current repository is strongest at:
- multimodal conditioning
- Amazon preprocessing
- DiT-based sequence-conditioned prediction
- distributed training infrastructure

The current repository is weakest at:
- discrete tokenization of items
- semantic ID / codebook support
- generative decoding
- constrained inference such as prefix-tree decoding

UniGenRec's modular structure is a good template because it separates:
- representation building
- semantic ID generation
- recommender training
- inference / evaluation

This matches the next step we want for this codebase.

## Current Repo -> GenRec Mapping

### 1. Representation / Preprocessing

Current building blocks:
- [preprocess_amazon.py](/e:/RoboticsDiffusionTransformer/preprocess_amazon.py)
- [scripts/generate_item_embeddings_from_meta.py](/e:/RoboticsDiffusionTransformer/scripts/generate_item_embeddings_from_meta.py)
- [train/dataset.py](/e:/RoboticsDiffusionTransformer/train/dataset.py)

GenRec-style role:
- keep Amazon preprocessing
- keep split generation
- extend outputs from continuous item embeddings to tokenization-ready
  sequence records

Planned output additions:
- `train.jsonl`, `valid.jsonl`, `test.jsonl` in a tokenization-friendly format
- per-item metadata table
- per-item dense representations before quantization
- per-item semantic IDs after quantization

### 2. Quantization / Semantic ID

Current gap:
- the repo has continuous `item_embeddings.npy`
- it does not yet produce semantic ID codebooks

Planned new layer:
- `genrec/tokenization/` for semantic ID builders
- `genrec/tokenization` will initially support a simple codebook pipeline based
  on current 128-d item embeddings

Recommended stage order:
1. start with deterministic residual/product quantization on existing
   `item_embeddings.npy`
2. produce `item_to_code.json`
3. only then plug codes into DiT training

This keeps the migration incremental and debuggable.

### 3. Generative Backbone

Current building blocks:
- [models/rdt_runner.py](/e:/RoboticsDiffusionTransformer/models/rdt_runner.py)
- [models/rdt/model.py](/e:/RoboticsDiffusionTransformer/models/rdt/model.py)
- [models/multimodal_encoder/condition_encoder.py](/e:/RoboticsDiffusionTransformer/models/multimodal_encoder/condition_encoder.py)

New role for DiT:
- use DiT as a sequence-level generative backbone
- feed tokenized history instead of only continuous target embedding regression
- predict masked semantic IDs or next-item semantic IDs

Recommended backbone formulation:
- token embedding:
  semantic ID tokens for history items
- optional side tokens:
  text summary tokens, image summary tokens, user/profile tokens
- position embedding:
  sequence order over tokenized history
- DiT hidden states:
  backbone over the tokenized sequence
- output head:
  logits over code vocabulary or item vocabulary

### 4. Training Objective

Current objective:
- diffusion denoising loss on target embedding
- optional ranking loss over trainable item latent table

Planned GenRec-style objectives:
- masked code prediction loss
- autoregressive next-item code prediction loss
- optional auxiliary diffusion / contrastive loss

Recommended first production target:
- cross-entropy over semantic ID tokens as the main loss
- keep diffusion-style loss only as an optional auxiliary branch

That lets the project become genuinely generative instead of continuing to be a
retrieval model with a diffusion-shaped head.

### 5. Inference / Decoding

Current inference:
- predict `[B, 1, 128]`
- retrieve nearest item from the item embedding library

Planned GenRec-style inference:
- construct `history tokens + [MASK]` or a decoder prompt
- decode semantic ID sequence with greedy / beam search
- optionally constrain decoding with a prefix trie built from legal item codes
- map semantic ID back to item ID

This follows the direction used by UniGenRec, where inference is a dedicated
module rather than a raw embedding nearest-neighbor lookup.

## Proposed New Repository Layout

```text
genrec/
  data/
  inference/
  models/
  tokenization/
  contracts.py

docs/
  genrec_dit_migration_design.md

old/
  legacy_recsys_dit/
```

## Minimal V1 Implementation Plan

### Phase 0: Preserve Current Working Pipeline

Status:
- completed via `old/legacy_recsys_dit/`

### Phase 1: Tokenization Data Spec

Add a new token-oriented dataset path with records like:

```json
{
  "user_id": 123,
  "history_item_ids": [12, 18, 99],
  "target_item_id": 314,
  "history_semantic_ids": [[14, 88, 2, 190], [7, 21, 54, 3], [19, 11, 9, 77]],
  "target_semantic_ids": [42, 16, 8, 201],
  "split": "train"
}
```

### Phase 2: Semantic ID Builder

Add semantic ID generation from `item_embeddings.npy`:
- input: dense item embeddings
- output:
  - `item_to_code.json`
  - `code_to_item.json`
  - quantizer stats / config

### Phase 3: DiT-as-Backbone Runner

Add a new runner, separate from `RDTRunner`:
- `GenRecDiTRunner`
- input: tokenized history batch
- output: token logits for masked or next semantic IDs

Do not overwrite the old `RDTRunner` path.

### Phase 4: Decode Module

Add:
- greedy decode
- beam search
- prefix-tree constrained decode

### Phase 5: Evaluation

Support two evaluation modes:
- semantic ID exact-match metrics
- item-level recommendation metrics such as Hit@K / NDCG@K / MRR@K

## Recommended Migration Principles

- Keep `old/legacy_recsys_dit/` untouched.
- Keep the current main working tree as the development base.
- Introduce the new GenRec stack in parallel instead of editing every old file
  in-place.
- Reuse existing preprocessing and multimodal encoders where they help.
- Replace continuous retrieval with token decoding only after the semantic ID
  path is stable.

## Concrete Module Ownership

### Keep and reuse

- Amazon preprocessing and split generation
- SigLIP / T5 loading and condition encoders
- Accelerate/DDP training entry patterns
- existing evaluation metric utilities where possible

### Build fresh

- semantic ID quantization layer
- token sequence dataset and collator
- DiT generative runner over token inputs
- decoding module
- GenRec-specific config files

## First Coding Targets After This Design

1. Add semantic ID generation from current `item_embeddings.npy`.
2. Add a tokenized dataset class parallel to the current continuous dataset.
3. Add `GenRecDiTRunner` with cross-entropy token prediction.
4. Add a decoding script that maps predicted semantic IDs back to items.

This sequence is the safest path from the current RecSys-DiT codebase to a
real GenRec-style pipeline.
