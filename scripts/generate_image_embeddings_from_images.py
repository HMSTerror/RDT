#!/usr/bin/env python
# coding=utf-8

"""
Generate image-based item embeddings aligned with the Amazon RecSys buffer.

This script:
1. reads the raw Amazon review file
2. applies the same iterative 5-core filtering used by preprocess_amazon.py
3. derives the exact item ordering expected by preprocess_amazon.py
4. encodes local item images with a local or remote SigLIP vision encoder
5. reduces the encoder output to 128 dimensions with PCA
6. writes item_embeddings_image.npy for downstream GenRec training/evaluation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess_amazon import (  # noqa: E402
    EMBED_DIM,
    K_CORE,
    apply_iterative_k_core,
    build_contiguous_mappings,
    load_review_interactions,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate image-based 128-d item_embeddings_image.npy aligned with "
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
        "--image-root",
        type=Path,
        default=Path("data/images"),
        help="Directory containing local item images named as {asin}.jpg / png / jpeg / webp.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/item_embeddings_image.npy"),
        help="Where to save the generated [num_items, 128] float32 image embedding matrix.",
    )
    parser.add_argument(
        "--vision-model-name-or-path",
        type=str,
        default="google/siglip-base-patch16-224",
        help="HF model id or local model directory used to encode item images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used when encoding item images.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=EMBED_DIM,
        help="Final embedding dimension. GenRec currently expects 128.",
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
        help="Torch dtype used during image encoding.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require that the vision encoder/processor are already available locally.",
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


def build_aligned_item_ids(reviews_path: Path) -> List[str]:
    interactions = load_review_interactions(reviews_path)
    if not interactions:
        raise RuntimeError("No valid interactions were found in the review file.")
    print(f"[reviews] raw interactions: {len(interactions)}")

    filtered = apply_iterative_k_core(interactions, min_count=K_CORE)
    if not filtered:
        raise RuntimeError("5-core filtering removed all interactions.")

    _, item_map = build_contiguous_mappings(filtered)
    ordered_item_ids = [item_id for item_id, _ in sorted(item_map.items(), key=lambda pair: pair[1])]
    print(f"[mapping] aligned items after 5-core filtering: {len(ordered_item_ids)}")
    return ordered_item_ids


def resolve_local_image_path(image_root: Path, item_id: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_root / f"{item_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_images_for_batch(
    image_root: Path,
    item_ids: Sequence[str],
) -> tuple[List[Image.Image], torch.Tensor]:
    images: List[Image.Image] = []
    missing_mask = torch.zeros(len(item_ids), dtype=torch.bool)

    for idx, item_id in enumerate(item_ids):
        image_path = resolve_local_image_path(image_root, item_id)
        if image_path is None:
            missing_mask[idx] = True
            images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
            continue

        try:
            with Image.open(image_path) as image:
                images.append(image.convert("RGB"))
        except Exception:
            missing_mask[idx] = True
            images.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))

    return images, missing_mask


def load_vision_backbone(
    model_name_or_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    local_files_only: bool,
):
    processor = SiglipImageProcessor.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )
    model = SiglipVisionModel.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).eval()
    model.to(device)
    return processor, model


@torch.no_grad()
def encode_images(
    item_ids: Sequence[str],
    *,
    image_root: Path,
    processor: SiglipImageProcessor,
    model: SiglipVisionModel,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    all_embeddings: List[torch.Tensor] = []
    missing_count = 0

    for start in tqdm(range(0, len(item_ids), batch_size), desc="Encoding item images", unit="batch"):
        batch_item_ids = list(item_ids[start : start + batch_size])
        images, missing_mask = load_images_for_batch(image_root, batch_item_ids)
        missing_count += int(missing_mask.sum().item())

        encoded = processor(images=images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device=device, dtype=model.dtype)
        outputs = model(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state.mean(dim=1).float()
        pooled = F.normalize(pooled, dim=-1)

        if missing_mask.any():
            pooled[missing_mask.to(device=pooled.device)] = 0.0
        all_embeddings.append(pooled.cpu())

    print(f"[images] missing_or_unreadable={missing_count}/{len(item_ids)}")
    return torch.cat(all_embeddings, dim=0)


def main() -> None:
    args = parse_args()

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file without --overwrite: {args.output_path}"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.image_root.mkdir(parents=True, exist_ok=True)

    ordered_item_ids = build_aligned_item_ids(args.reviews_path)
    processor, model = load_vision_backbone(
        args.vision_model_name_or_path,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    embeddings = encode_images(
        ordered_item_ids,
        image_root=args.image_root,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    reduced = reduce_to_target_dim(embeddings, output_dim=args.output_dim, seed=args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, reduced.cpu().numpy().astype(np.float32, copy=False))

    sidecar_path = args.output_path.with_suffix(".meta.json")
    payload = {
        "reviews_path": str(args.reviews_path),
        "image_root": str(args.image_root),
        "output_path": str(args.output_path),
        "vision_model_name_or_path": args.vision_model_name_or_path,
        "num_items": int(len(ordered_item_ids)),
        "output_dim": int(args.output_dim),
        "dtype": str(dtype),
        "local_files_only": bool(args.local_files_only),
    }
    with open(sidecar_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print("[done] wrote", tuple(reduced.shape), "float32 embeddings to", args.output_path)
    print("[done] wrote sidecar metadata to", sidecar_path)


if __name__ == "__main__":
    main()
