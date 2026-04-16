# GenRec Workspace

This package hosts the active GenRec-style recommendation stack used in the repository.

Current scope:

- raw Amazon Music data helpers
- text / image / CF embedding preparation
- multimodal fusion utilities
- semantic ID quantization
- tokenized train / val / test datasets
- DiT-style semantic token backbone
- hybrid diffusion over continuous target item latents
- grouped retrieval evaluation for hot / mid / cold items

Key modules:

- [genrec/raw_data/amazon_music.py](/e:/RoboticsDiffusionTransformer/genrec/raw_data/amazon_music.py)
- [genrec/quantization/](/e:/RoboticsDiffusionTransformer/genrec/quantization)
- [genrec/tokenization/semantic_ids.py](/e:/RoboticsDiffusionTransformer/genrec/tokenization/semantic_ids.py)
- [genrec/data/tokenized_dataset.py](/e:/RoboticsDiffusionTransformer/genrec/data/tokenized_dataset.py)
- [genrec/models/genrec_dit.py](/e:/RoboticsDiffusionTransformer/genrec/models/genrec_dit.py)
- [genrec/models/genrec_hybrid_diffusion.py](/e:/RoboticsDiffusionTransformer/genrec/models/genrec_hybrid_diffusion.py)
- [genrec/inference/semantic_decoder.py](/e:/RoboticsDiffusionTransformer/genrec/inference/semantic_decoder.py)

Main entry points:

- [scripts/prepare_genrec_semantic_ids.sh](/e:/RoboticsDiffusionTransformer/scripts/prepare_genrec_semantic_ids.sh)
- [scripts/train_genrec_hybrid_diffusion.py](/e:/RoboticsDiffusionTransformer/scripts/train_genrec_hybrid_diffusion.py)
- [scripts/eval_genrec_hybrid_diffusion.py](/e:/RoboticsDiffusionTransformer/scripts/eval_genrec_hybrid_diffusion.py)
- [scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh](/e:/RoboticsDiffusionTransformer/scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh)
- [scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh](/e:/RoboticsDiffusionTransformer/scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh)

Hybrid diffusion quick start:

```bash
bash scripts/run_genrec_hybrid_diffusion_train50k_clean.sh
bash scripts/run_genrec_hybrid_diffusion_eval.sh
```

Current recommended experiment entry:

```bash
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh
```

For from-scratch stage-2 experiments, the paired pipeline is:

```bash
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_fromscratch_pipeline.sh
```

By default this from-scratch pipeline skips a separate final full evaluation and relies on the `full` run inside the ablation suite. Set `RUN_FINAL_FULL_EVAL=1` to restore the extra standalone full evaluation.

The semantic-ID preparation pipeline now supports a stricter protocol for behavior-derived features. In `scripts/prepare_genrec_semantic_ids.sh`, the defaults are `ITEM_UNIVERSE_SPLIT=train`, `CF_FIT_SPLIT=train`, and `SEMANTIC_ID_FIT_SPLIT=train`, meaning the item universe can be restricted to train-prefix items, CF embeddings are fitted on the train prefix only, and the semantic-ID quantizer is fitted on train-seen items before encoding the item table. Set those variables to `all` if you explicitly want the older behavior.

The trainer also supports a small fixed validation subset plus storage-efficient early stopping. You can set `training.val_subset_size`, `training.val_subset_offset`, and `training.early_stopping` in the YAML config, then keep periodic checkpoint saving disabled so only `best/` and `final/` snapshots are written.

The stage-2 eval and ablation wrappers also accept `NUM_PROCESSES` and `MIXED_PRECISION`, so on a dual-GPU server you can launch evaluation with `NUM_PROCESSES=2 MIXED_PRECISION=bf16` and use both cards for the final full run and follow-up ablations.
