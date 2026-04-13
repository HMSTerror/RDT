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

Hybrid diffusion quick start:

```bash
bash scripts/run_genrec_hybrid_diffusion_train50k_clean.sh
bash scripts/run_genrec_hybrid_diffusion_eval.sh
```

Current recommended experiment entry:

```bash
bash scripts/run_genrec_hybrid_diffusion_stage2_fullmodal_30ep_pipeline.sh
```
