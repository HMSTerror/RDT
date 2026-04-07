#!/usr/bin/env python
# coding=utf-8

"""
Generate metadata-based item embeddings aligned with the Amazon RecSys buffer.

This script:
1. reads the raw Amazon review file
2. applies the same iterative 5-core filtering used by preprocess_amazon.py
3. derives the exact item ordering expected by preprocess_amazon.py
4. encodes item metadata text with a local or remote text encoder
5. reduces the encoder output to 128 dimensions with PCA
6. writes item_embeddings.npy for downstream preprocessing/training

The resulting `item_embeddings.npy` is a semantic text embedding matrix rather
than the dummy random matrix used for smoke testing. It is "real" in the sense
that each item vector is computed from its actual metadata, but it is still a
metadata-derived representation rather than a collaborative embedding learned
from interaction sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess_amazon import (  # noqa: E402
    EMBED_DIM,
    K_CORE,
    apply_iterative_k_core,
    build_contiguous_mappings,
    clean_text,
    format_categories,
    iter_json_records,
    load_review_interactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate metadata-based 128-d item_embeddings.npy aligned with "
            "preprocess_amazon.py item ordering."
        )
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/Musical_Instruments_5.json"),
        help="Path to the raw Amazon review JSON / JSON.GZ file.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json"),
        help="Path to the raw Amazon metadata JSON / JSON.GZ file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/item_embeddings.npy"),
        help="Where to save the generated [num_items, 128] float32 embedding matrix.",
    )
    parser.add_argument(
        "--text-model-name-or-path",
        type=str,
        default="google/t5-v1_1-small",
        help="HF model id or local model directory used to encode item metadata text.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when encoding metadata text.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=96,
        help="Tokenizer max sequence length for item metadata text.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=EMBED_DIM,
        help="Final embedding dimension. RecSys-DiT expects 128.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda, cuda:0, or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Torch dtype used during text encoding.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require that the text encoder/tokenizer are already available locally.",
    )
    parser.add_argument(
        "--max-description-chars",
        type=int,
        default=256,
        help="Maximum number of description characters retained per item.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file if it already exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used by PCA.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "fp32":
        return torch.float32
    if dtype_arg == "fp16":
        return torch.float16
    if dtype_arg == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_item_text(record: dict, *, item_id: str, max_description_chars: int) -> str:
    title = clean_text(record.get("title", ""))
    brand = clean_text(record.get("brand", ""))
    categories = format_categories(record.get("categories"))
    description = truncate_text(
        clean_text(record.get("description", "")),
        max_description_chars,
    )
    feature = truncate_text(
        clean_text(record.get("feature", "")),
        max_description_chars,
    )

    fields: List[str] = []
    if title:
        fields.append(f"title: {title}")
    if brand:
        fields.append(f"brand: {brand}")
    if categories:
        fields.append(f"categories: {categories}")
    if description:
        fields.append(f"description: {description}")
    if feature:
        fields.append(f"feature: {feature}")

    return " ; ".join(fields) or item_id


def load_item_texts(
    meta_path: Path,
    ordered_item_ids: Sequence[str],
    *,
    max_description_chars: int,
) -> List[str]:
    target_set = set(ordered_item_ids)
    item_text_map: Dict[str, str] = {}

    for record in iter_json_records(meta_path, desc="Reading metadata for item embeddings"):
        item_id = clean_text(record.get("asin", ""))
        if not item_id or item_id not in target_set:
            continue
        item_text_map[item_id] = build_item_text(
            record,
            item_id=item_id,
            max_description_chars=max_description_chars,
        )
        if len(item_text_map) == len(target_set):
            break

    texts: List[str] = []
    missing_meta = 0
    for item_id in ordered_item_ids:
        text = item_text_map.get(item_id)
        if text is None:
            missing_meta += 1
            text = item_id
        texts.append(text)

    print(
        f"[metadata] loaded text for {len(item_text_map)}/{len(ordered_item_ids)} items; "
        f"missing_meta={missing_meta}"
    )
    return texts


def load_text_backbone(
    model_name_or_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    model_cls = T5EncoderModel if getattr(config, "model_type", "") == "t5" else AutoModel
    model = model_cls.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).eval()
    model.to(device)
    return tokenizer, model


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    *,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    all_embeddings: List[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding item metadata", unit="batch"):
        batch_text = list(texts[start : start + batch_size])
        tokens = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        attention_mask = tokens["attention_mask"].to(device)
        input_ids = tokens["input_ids"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = outputs.last_hidden_state
        pooled = mean_pool(hidden_state, attention_mask)
        pooled = F.normalize(pooled.float(), dim=-1)
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def reduce_to_target_dim(embeddings: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    num_items, raw_dim = embeddings.shape
    if raw_dim == output_dim:
        return F.normalize(embeddings, dim=-1)

    if raw_dim < output_dim:
        padded = torch.zeros(num_items, output_dim, dtype=embeddings.dtype)
        padded[:, :raw_dim] = embeddings
        return F.normalize(padded, dim=-1)

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    max_rank = min(num_items, raw_dim)
    effective_dim = min(output_dim, max_rank)

    torch.manual_seed(seed)
    q = min(max(effective_dim + 8, effective_dim), max_rank)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    reduced = centered @ v[:, :effective_dim]

    if effective_dim < output_dim:
        padded = torch.zeros(num_items, output_dim, dtype=reduced.dtype)
        padded[:, :effective_dim] = reduced
        reduced = padded

    return F.normalize(reduced, dim=-1)


def save_sidecar_metadata(
    output_path: Path,
    *,
    args: argparse.Namespace,
    num_items: int,
    raw_dim: int,
) -> None:
    meta_payload = {
        "reviews_path": str(args.reviews_path),
        "meta_path": str(args.meta_path),
        "output_path": str(output_path),
        "text_model_name_or_path": args.text_model_name_or_path,
        "local_files_only": bool(args.local_files_only),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "output_dim": int(args.output_dim),
        "raw_encoder_dim": int(raw_dim),
        "num_items": int(num_items),
        "k_core": int(K_CORE),
        "embedding_type": "metadata_text_mean_pool_then_pca",
    }
    sidecar_path = output_path.with_suffix(".meta.json")
    with open(sidecar_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    if not args.reviews_path.exists():
        raise FileNotFoundError(f"Review file not found: {args.reviews_path}")
    if not args.meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {args.meta_path}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_path}. "
            "Pass --overwrite to replace it."
        )

    print("========== Generate Item Embeddings ==========")
    print(f"reviews_path          : {args.reviews_path}")
    print(f"meta_path             : {args.meta_path}")
    print(f"output_path           : {args.output_path}")
    print(f"text_model            : {args.text_model_name_or_path}")
    print(f"device                : {device}")
    print(f"dtype                 : {dtype}")
    print(f"local_files_only      : {bool(args.local_files_only)}")

    interactions = load_review_interactions(args.reviews_path)
    if not interactions:
        raise RuntimeError("No valid interactions were found in the review file.")
    print(f"[reviews] raw interactions: {len(interactions)}")

    filtered = apply_iterative_k_core(interactions, min_count=K_CORE)
    if not filtered:
        raise RuntimeError("5-core filtering removed all interactions.")

    _, item_map = build_contiguous_mappings(filtered)
    ordered_item_ids = [
        item_id for item_id, _ in sorted(item_map.items(), key=lambda pair: pair[1])
    ]
    print(f"[mapping] aligned items after {K_CORE}-core filtering: {len(ordered_item_ids)}")

    item_texts = load_item_texts(
        args.meta_path,
        ordered_item_ids,
        max_description_chars=args.max_description_chars,
    )

    tokenizer, model = load_text_backbone(
        args.text_model_name_or_path,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    raw_embeddings = encode_texts(
        item_texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    reduced_embeddings = reduce_to_target_dim(
        raw_embeddings,
        output_dim=args.output_dim,
        seed=args.seed,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_path,
        reduced_embeddings.cpu().numpy().astype(np.float32, copy=False),
    )
    save_sidecar_metadata(
        args.output_path,
        args=args,
        num_items=reduced_embeddings.shape[0],
        raw_dim=raw_embeddings.shape[1],
    )

    print(
        f"[done] wrote {tuple(reduced_embeddings.shape)} float32 embeddings to {args.output_path}"
    )
    print(f"[done] wrote sidecar metadata to {args.output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
