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
import ast
import io
import json
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

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
    build_item_map_from_sequences,
    build_user_sequences,
    extract_item_id,
    load_review_interactions,
    select_item_universe_sequences,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def parse_json_record(line: str) -> Optional[dict]:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return None


def iter_json_records(path: Path, desc: str) -> Iterator[dict]:
    with open_text(path) as handle:
        for line in tqdm(handle, desc=desc, unit="line"):
            record = parse_json_record(line)
            if record is not None:
                yield record


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x)
    return " ".join(str(value).split())


def choose_image_url(meta: dict) -> str:
    candidates: List[str] = []
    for key in ("imUrl", "imageURL", "imageURLHighRes"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        elif isinstance(value, list):
            candidates.extend([str(x).strip() for x in value if str(x).strip()])
    images = meta.get("images")
    if isinstance(images, dict):
        for key in ("hi_res", "large", "thumb"):
            value = images.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
            elif isinstance(value, list):
                candidates.extend([str(x).strip() for x in value if str(x).strip() and str(x).strip() != "None"])
    return candidates[0] if candidates else ""


def filename_from_url(image_url: str) -> str:
    if not image_url:
        return ""
    return image_url.split("?")[0].rsplit("/", 1)[-1].strip()


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
        "--meta-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json"),
        help="Path to metadata JSON / JSON.GZ used for image URL and filename hints.",
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
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Try downloading missing images from metadata URLs before encoding.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Thread workers used when --download-missing is enabled.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=10.0,
        help="Per-request timeout in seconds when downloading missing images.",
    )
    parser.add_argument(
        "--allow-all-missing",
        action="store_true",
        help="Allow output even when all items are missing images.",
    )
    parser.add_argument(
        "--item-universe-split",
        type=str,
        default="all",
        choices=["all", "train"],
        help=(
            "Which temporal split defines the aligned item ordering. "
            "`train` uses only the leave-last-two train prefix items."
        ),
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="leave_last_two",
        choices=["none", "leave_last_two"],
        help="Temporal split mode paired with --item-universe-split.",
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


def build_aligned_item_ids(
    reviews_path: Path,
    *,
    item_universe_split: str,
    split_mode: str,
) -> List[str]:
    interactions = load_review_interactions(reviews_path)
    if not interactions:
        raise RuntimeError("No valid interactions were found in the review file.")
    print(f"[reviews] raw interactions: {len(interactions)}")

    filtered = apply_iterative_k_core(interactions, min_count=K_CORE)
    if not filtered:
        raise RuntimeError("5-core filtering removed all interactions.")

    user_map, _ = build_contiguous_mappings(filtered)
    sequences = build_user_sequences(filtered, user_map)
    universe_sequences = select_item_universe_sequences(
        sequences,
        item_universe_split=item_universe_split,
        split_mode=split_mode,
    )
    item_map = build_item_map_from_sequences(universe_sequences)
    if not item_map:
        raise RuntimeError("The selected item universe is empty after temporal filtering.")
    ordered_item_ids = [item_id for item_id, _ in sorted(item_map.items(), key=lambda pair: pair[1])]
    print(
        f"[mapping] aligned items after 5-core filtering: {len(ordered_item_ids)} "
        f"(item_universe_split={item_universe_split})"
    )
    return ordered_item_ids


def load_meta_image_lookup(
    meta_path: Path,
    target_item_ids: Sequence[str],
) -> Dict[str, dict]:
    target_set = set(target_item_ids)
    lookup: Dict[str, dict] = {}
    for record in iter_json_records(meta_path, desc="Reading metadata for image lookup"):
        item_id = extract_item_id(record)
        if not item_id or item_id not in target_set:
            continue
        image_url = choose_image_url(record)
        image_filename = filename_from_url(image_url)
        image_path = clean_text(record.get("image_path", ""))
        lookup[item_id] = {
            "image_url": image_url,
            "image_filename": image_filename,
            "image_path": image_path,
        }
        if len(lookup) == len(target_set):
            break
    print(f"[metadata] image hints for {len(lookup)}/{len(target_item_ids)} aligned items")
    return lookup


def build_recursive_image_index(image_root: Path) -> tuple[Dict[str, Path], Dict[str, Path], int]:
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    num_image_files = 0
    if not image_root.exists():
        return by_name, by_stem, num_image_files

    for path in image_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        num_image_files += 1
        name_key = path.name.lower()
        stem_key = path.stem.lower()
        if name_key not in by_name:
            by_name[name_key] = path
        if stem_key not in by_stem:
            by_stem[stem_key] = path

    return by_name, by_stem, num_image_files


def resolve_local_image_path(
    image_root: Path,
    item_id: str,
    *,
    meta_lookup: Dict[str, dict],
    index_by_name: Dict[str, Path],
    index_by_stem: Dict[str, Path],
) -> Path | None:
    candidates: List[Path] = []

    for suffix in IMAGE_EXTENSIONS:
        candidates.append(image_root / f"{item_id}{suffix}")

    meta = meta_lookup.get(item_id, {})
    image_path = clean_text(meta.get("image_path", ""))
    image_filename = clean_text(meta.get("image_filename", ""))

    if image_path:
        candidates.append(image_root / image_path)
    if image_filename:
        candidates.append(image_root / image_filename)
        candidates.append(image_root / item_id / image_filename)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    if image_filename:
        candidate = index_by_name.get(image_filename.lower())
        if candidate is not None and candidate.exists():
            return candidate

    candidate = index_by_stem.get(item_id.lower())
    if candidate is not None and candidate.exists():
        return candidate

    return None


def _download_one_image(
    item_id: str,
    *,
    meta_lookup: Dict[str, dict],
    image_root: Path,
    timeout: float,
) -> bool:
    meta = meta_lookup.get(item_id, {})
    image_url = clean_text(meta.get("image_url", ""))
    if not image_url:
        return False

    output_path = image_root / f"{item_id}.jpg"
    if output_path.exists():
        return True

    request = urllib.request.Request(
        image_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; GenRec-Image-Embedding/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = response.read()
        with Image.open(io.BytesIO(payload)) as image:
            image = image.convert("RGB")
            image.save(output_path, format="JPEG", quality=95)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def prefetch_missing_images(
    item_ids: Sequence[str],
    *,
    image_root: Path,
    meta_lookup: Dict[str, dict],
    index_by_name: Dict[str, Path],
    index_by_stem: Dict[str, Path],
    num_workers: int,
    timeout: float,
) -> tuple[int, int]:
    pending: List[str] = []
    for item_id in item_ids:
        if resolve_local_image_path(
            image_root,
            item_id,
            meta_lookup=meta_lookup,
            index_by_name=index_by_name,
            index_by_stem=index_by_stem,
        ) is not None:
            continue
        if clean_text(meta_lookup.get(item_id, {}).get("image_url", "")):
            pending.append(item_id)

    if not pending:
        print("[images] no downloadable missing items found")
        return 0, 0

    print(
        f"[images] prefetch missing: pending={len(pending)} workers={max(1, int(num_workers))} timeout={timeout}s"
    )
    success = 0
    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        futures = {
            executor.submit(
                _download_one_image,
                item_id,
                meta_lookup=meta_lookup,
                image_root=image_root,
                timeout=timeout,
            ): item_id
            for item_id in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images", unit="img"):
            if future.result():
                success += 1
    return len(pending), success


def load_images_for_batch(
    image_root: Path,
    item_ids: Sequence[str],
    *,
    meta_lookup: Dict[str, dict],
    index_by_name: Dict[str, Path],
    index_by_stem: Dict[str, Path],
) -> tuple[List[Image.Image], torch.Tensor]:
    images: List[Image.Image] = []
    missing_mask = torch.zeros(len(item_ids), dtype=torch.bool)

    for idx, item_id in enumerate(item_ids):
        image_path = resolve_local_image_path(
            image_root,
            item_id,
            meta_lookup=meta_lookup,
            index_by_name=index_by_name,
            index_by_stem=index_by_stem,
        )
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
    meta_lookup: Dict[str, dict],
    index_by_name: Dict[str, Path],
    index_by_stem: Dict[str, Path],
    processor: SiglipImageProcessor,
    model: SiglipVisionModel,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, int]:
    all_embeddings: List[torch.Tensor] = []
    missing_count = 0

    for start in tqdm(range(0, len(item_ids), batch_size), desc="Encoding item images", unit="batch"):
        batch_item_ids = list(item_ids[start : start + batch_size])
        images, missing_mask = load_images_for_batch(
            image_root,
            batch_item_ids,
            meta_lookup=meta_lookup,
            index_by_name=index_by_name,
            index_by_stem=index_by_stem,
        )
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
    return torch.cat(all_embeddings, dim=0), missing_count


def main() -> None:
    args = parse_args()

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output file without --overwrite: {args.output_path}"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.image_root.mkdir(parents=True, exist_ok=True)

    print(f"item_universe_split    : {args.item_universe_split}")
    print(f"split_mode             : {args.split_mode}")

    ordered_item_ids = build_aligned_item_ids(
        args.reviews_path,
        item_universe_split=args.item_universe_split,
        split_mode=args.split_mode,
    )
    meta_lookup = load_meta_image_lookup(args.meta_path, ordered_item_ids)
    index_by_name, index_by_stem, indexed_files = build_recursive_image_index(args.image_root)
    print(
        f"[images] indexed local files under {args.image_root}: "
        f"{indexed_files} (name_index={len(index_by_name)}, stem_index={len(index_by_stem)})"
    )

    if args.download_missing:
        pending, success = prefetch_missing_images(
            ordered_item_ids,
            image_root=args.image_root,
            meta_lookup=meta_lookup,
            index_by_name=index_by_name,
            index_by_stem=index_by_stem,
            num_workers=args.download_workers,
            timeout=args.download_timeout,
        )
        if pending > 0:
            print(f"[images] download success={success}/{pending}")
            index_by_name, index_by_stem, indexed_files = build_recursive_image_index(args.image_root)
            print(
                f"[images] re-indexed local files: "
                f"{indexed_files} (name_index={len(index_by_name)}, stem_index={len(index_by_stem)})"
            )

    processor, model = load_vision_backbone(
        args.vision_model_name_or_path,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    embeddings, missing_count = encode_images(
        ordered_item_ids,
        image_root=args.image_root,
        meta_lookup=meta_lookup,
        index_by_name=index_by_name,
        index_by_stem=index_by_stem,
        processor=processor,
        model=model,
        device=device,
        batch_size=args.batch_size,
    )
    if missing_count == len(ordered_item_ids) and not args.allow_all_missing:
        raise RuntimeError(
            "All items are missing/unreadable images. "
            "Check IMAGE_ROOT, file naming, and metadata URL availability; "
            "or rerun with --download-missing. "
            "If you intentionally want zero-image embeddings, pass --allow-all-missing."
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
        "missing_or_unreadable": int(missing_count),
        "output_dim": int(args.output_dim),
        "dtype": str(dtype),
        "local_files_only": bool(args.local_files_only),
        "download_missing": bool(args.download_missing),
        "item_universe_split": args.item_universe_split,
        "split_mode": args.split_mode,
    }
    with open(sidecar_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print("[done] wrote", tuple(reduced.shape), "float32 embeddings to", args.output_path)
    print("[done] wrote sidecar metadata to", sidecar_path)


if __name__ == "__main__":
    main()
