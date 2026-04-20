# V1.4 Content-Branch Status Report

## 1. Background

`v1.4` switched the main recommendation space to a cleaner CF-only formulation:

- main sequence: `CF-only semantic-ID history`
- diffusion target space: `item_embeddings_cf.npy`
- retrieval table: `item_embeddings_cf.npy`
- side branches: `text` and `image` via cross-attention

This made the earlier conclusion much cleaner: if `text/image` matter, they must now help from the side branch rather than leaking through fused semantic IDs.

## 2. Three Auxiliary-Retrieval Variants

### A. Mainline pooled-only auxiliary retrieval

Config:

- [genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval.yaml](/e:/RDT/configs/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval.yaml)

Key settings:

- `text_weight = 0.1`
- `image_weight = 0.1`
- `text_history_weight = 0.0`
- `image_history_weight = 0.0`
- no curriculum

Observed behavior:

- best overall among the tested auxiliary-retrieval variants
- clear modality sensitivity:
  - removing `text` causes a large drop
  - removing `image` causes a large drop

Interpretation:

- pooled auxiliary retrieval is strong enough to make `text/image` useful
- but still restrained enough not to derail the CF backbone

### B. Full history-aux + curriculum variant

Config:

- [genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval_histaux.yaml](/e:/RDT/configs/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval_histaux.yaml)

Key settings:

- `text_weight = image_weight = 0.15`
- `text_history_weight = image_history_weight = 0.15`
- curriculum:
  - epochs `1-5`: `0.3`
  - epochs `6+`: `0.15`

Observed behavior:

- overall metric did not improve over pooled-only
- ablation sensitivity became much weaker

Interpretation:

- this version likely trained the side auxiliary heads more than it improved the main denoising/retrieval path
- history-summary supervision appears too coarse for next-item recommendation in the current form
- the strong early curriculum likely encourages the branches to solve their own retrieval objective instead of improving the main branch-to-backbone interaction

### C. Light history regularizer

Config:

- [genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval_light_histreg.yaml](/e:/RDT/configs/genrec_hybrid_diffusion_amazon_stage2_fullmodal_30ep_v14_cf_only_main_aligned_lowdrop_auxretrieval_light_histreg.yaml)

Key settings:

- `text_weight = image_weight = 0.1`
- `text_history_weight = image_history_weight = 0.02`
- no curriculum

Intended role:

- preserve the successful pooled-only mainline
- add only a weak history-side regularization signal
- test whether history supervision can help as a gentle bias instead of a co-equal objective

## 3. Why We Are Reverting the Mainline

The pooled-only version should remain the mainline for now because it offers the best trade-off:

- highest or near-highest overall quality
- strong evidence that `text/image` are genuinely used
- simpler explanation
- lower risk of over-optimizing auxiliary branches at the expense of the core CF diffusion objective

The history-aux version is still valuable as a negative result:

- it shows that stronger auxiliary supervision is not automatically better
- it suggests that naive history-summary retrieval supervision may be misaligned with next-item ranking

## 4. Current Recommendation

Recommended mainline:

- pooled-only auxiliary retrieval (`0.1 / 0.1`)

Recommended next experiment:

- run the new light history regularizer version
- compare against pooled-only on:
  - overall `ndcg@20`
  - `no_text`
  - `no_image`
  - grouped cold/mid/hot metrics

## 5. High-Level Conclusion

The project is now in a much clearer stage than before:

- `text/image` can be learned and can affect recommendation quality
- the key problem is no longer "content branches are ignored"
- the current problem is "how much auxiliary pressure is helpful before it stops improving the main task"

At this stage, the evidence supports a conservative content strategy:

- strong pooled supervision
- weak or no history-level regularization
- avoid aggressive early curriculum unless a more target-aware history aggregation mechanism is introduced later
