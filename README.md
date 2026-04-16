# GenRec Hybrid Diffusion

This repository now centers on a GenRec-style hybrid diffusion recommendation pipeline:

- raw Amazon review data
- text / image / CF item embeddings
- optional multimodal fusion
- semantic ID quantization and tokenized train/val/test samples
- hybrid diffusion training over continuous target latents
- grouped retrieval evaluation on hot / mid / cold items

The active code path is built around:

- [preprocess_amazon.py](/e:/RoboticsDiffusionTransformer/preprocess_amazon.py)
- [genrec/](/e:/RoboticsDiffusionTransformer/genrec)
- [scripts/prepare_genrec_semantic_ids.sh](/e:/RoboticsDiffusionTransformer/scripts/prepare_genrec_semantic_ids.sh)
- [scripts/train_genrec_hybrid_diffusion.py](/e:/RoboticsDiffusionTransformer/scripts/train_genrec_hybrid_diffusion.py)
- [scripts/eval_genrec_hybrid_diffusion.py](/e:/RoboticsDiffusionTransformer/scripts/eval_genrec_hybrid_diffusion.py)
- [scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh](/e:/RoboticsDiffusionTransformer/scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh)
- [scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh](/e:/RoboticsDiffusionTransformer/scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh)

## Install

```bash
# install torch / torchvision for your CUDA version first
pip install -r requirements.txt
pip install -r requirements_data.txt
```

## End-to-End Flow

1. Download the raw Amazon Musical Instruments data:

```bash
bash scripts/download_amazon_music_dataset.sh
```

2. Build multimodal dense item embeddings, semantic IDs, and tokenized splits:

```bash
TEXT_MODEL_NAME_OR_PATH=/path/to/t5 \
VISION_MODEL_NAME_OR_PATH=/path/to/siglip \
bash scripts/prepare_genrec_semantic_ids.sh

# strict protocol defaults in v1.3:
# - text / image / CF / preprocess item universe can all be restricted to the train prefix
# - CF embeddings fit on the train prefix only
# - semantic ID quantizer fits on train-seen items only, then encodes all items
```

3. Train the baseline hybrid diffusion model:

```bash
 bash scripts/run_genrec_hybrid_diffusion_train50k_clean.sh
 ```

4. Evaluate on the grouped test split:

```bash
bash scripts/run_genrec_hybrid_diffusion_eval.sh
```

5. For the current full-modal stage-2 workflow, run:

```bash
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh
```

6. For current from-scratch stage-2 experiments with automatic post-training ablations, run:

```bash
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh
```

## Main Outputs

- dense embeddings:
  `item_embeddings_text.npy`, `item_embeddings_image.npy`, `item_embeddings_cf.npy`, `item_embeddings_fused.npy`
- semantic IDs:
  `data/Amazon_Music_And_Instruments/semantic_ids_pq/`
- tokenized train/val/test splits:
  `data/Amazon_Music_And_Instruments/tokenized_semantic_ids/`
- hybrid diffusion checkpoints:
  `checkpoints/genrec_hybrid_diffusion_amazon_50k/`
- grouped evaluation artifacts:
  `logs/genrec_hybrid_*`, `logs/genrec_stage2_fullmodal_*`

## Notes

- `preprocess_amazon.py` is still part of the active pipeline because semantic-ID preparation depends on its split-aware buffer generation.
- `scripts/prepare_genrec_semantic_ids.sh` now defaults to a stricter protocol: `ITEM_UNIVERSE_SPLIT=train`, `CF_FIT_SPLIT=train`, and `SEMANTIC_ID_FIT_SPLIT=train`. This means the item universe can be restricted to train-prefix items, CF is fitted on train interactions, and semantic-ID fitting uses train-seen items only. Set those variables to `all` if you explicitly want the older behavior.
- Training now supports a storage-efficient early-stopping mode driven by a fixed small validation subset. Add `training.val_subset_size`, `training.val_subset_offset`, and `training.early_stopping.{enabled,metric,mode,patience,min_delta,warmup_epochs}` to the YAML config, and keep `save_every_epochs=0` / `save_steps=0` if you want only `best/` and `final/` snapshots.
- The evaluation script reports both overall retrieval metrics and grouped long-tail metrics on `cold`, `mid`, and `hot` items.
- `scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh` now skips a separate final full evaluation by default and goes directly to the ablation suite, because the ablation suite already includes a `full` run. Set `RUN_FINAL_FULL_EVAL=1` if you want both.
- Stage-2 eval and ablation wrappers now support distributed `accelerate launch`. On a dual-GPU server you can set `NUM_PROCESSES=2 MIXED_PRECISION=bf16` so the final full eval and ablation runs use both cards instead of a single process.
- [genrec/models/genrec_dit.py](/e:/RoboticsDiffusionTransformer/genrec/models/genrec_dit.py) is retained as the shared semantic-token backbone block used by the hybrid diffusion model, not as a separate maintained training pipeline.
