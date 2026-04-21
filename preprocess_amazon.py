#!/usr/bin/env python
# coding=utf-8

"""
Offline preprocessing for the Amazon_Music_And_Instruments dataset.

This script converts raw Amazon review logs and metadata into a lightweight
"chunk index + global item store" format for RecSys-DiT:

- buffer/chunk_0/dirty_bit
- buffer/chunk_0/samples.npz
- buffer/item_embeddings.npy
- buffer/item_meta.json
- buffer/item_map.json
- buffer/user_map.json
- buffer/stats.json

Instead of duplicating image tensors and embeddings into every sample, each
sample stores only:
- user_id
- history_item_ids [HISTORY_LEN]
- history_mask [HISTORY_LEN]
- target_item_id

The dataset reconstructs embeddings and images dynamically from the global
item store at training time. This keeps the preprocessing output practical in
both disk usage and I/O.

Important implementation note:
To keep the chunk layout compatible with fixed-size dirty_bit buffers, this
script drops the final incomplete chunk by default. This avoids generating a
dirty_bit file of length CHUNK_SIZE for a chunk that contains fewer than
CHUNK_SIZE actual samples.
"""

from __future__ import annotations

import argparse
import ast
import gc
import gzip
import io
import json
import math
import os
import shutil
import urllib.error
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from data.filelock import FileLock


HISTORY_LEN = 50
CHUNK_SIZE = 1000
K_CORE = 5
EMBED_DIM = 128
IMAGE_SIZE = 224
DATASET_NAME = "amazon_music"
IMAGE_CACHE_SIZE = 256
SPLIT_NAMES = ("train", "val", "test")
SUPPORTED_ITEM_UNIVERSE_SPLITS = ("all", "train")

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    RESAMPLE_BICUBIC = Image.BICUBIC


IMAGE_STORE: "ImageStore | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon_Music_And_Instruments into RecSys-DiT chunks."
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/Musical_Instruments_5.json"),
        help="Path to raw review JSON/JSON.GZ file.",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json"),
        help="Path to raw metadata JSON/JSON.GZ file.",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/item_embeddings.npy"),
        help="Path to the pre-trained item embedding matrix [num_items, 128].",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data/images"),
        help="Local image directory. Images are stored as {item_id}.jpg.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("buffer"),
        help="Output directory for chunk_{i} folders.",
    )
    parser.add_argument(
        "--history-len",
        type=int,
        default=HISTORY_LEN,
        help="History length for sliding-window sequence truncation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of samples per output chunk.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Number of threads used for image download fallback.",
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for each image download request.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Disable URL-based image download fallback and use only local images.",
    )
    parser.add_argument(
        "--prefetch-images",
        action="store_true",
        help="Optionally prefetch missing local images from metadata URLs into image_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing new chunks.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on the number of written samples. Rounded down to a multiple of chunk_size.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute dataset statistics and storage estimates, without writing chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used only for dummy item embeddings.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="leave_last_two",
        choices=["none", "leave_last_two"],
        help=(
            "How to split user sequences into buffers. "
            "`leave_last_two` writes self-contained train/val/test buffers where "
            "each user's last target is test, the previous target is val, and all "
            "earlier targets are train. Use `none` to keep the legacy single-buffer layout."
        ),
    )
    parser.add_argument(
        "--item-universe-split",
        type=str,
        default="all",
        choices=list(SUPPORTED_ITEM_UNIVERSE_SPLITS),
        help=(
            "Which temporal split defines the item universe and item_map. "
            "`all` preserves the legacy behavior. "
            "`train` builds the item universe from the train prefix only under "
            "the chosen --split-mode, dropping val/test-only target items."
        ),
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
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


def extract_item_id(record: dict) -> str:
    """
    Prefer the stable product identifier used by the dataset version.

    Amazon Reviews'23 uses `parent_asin` as the canonical product id for
    matching reviews with metadata. Older Amazon dumps typically only expose
    `asin`.
    """
    return clean_text(record.get("parent_asin", "")) or clean_text(record.get("asin", ""))


def extract_user_id(record: dict) -> str:
    """
    Support both legacy Amazon reviewer ids and Amazon Reviews'23 user ids.
    """
    return clean_text(record.get("user_id", "")) or clean_text(record.get("reviewerID", ""))


def extract_timestamp(record: dict) -> int:
    """
    Support both legacy unixReviewTime and Amazon Reviews'23 millisecond
    timestamps.
    """
    for key in ("timestamp", "sort_timestamp", "unixReviewTime"):
        value = record.get(key, 0)
        try:
            if value is None or value == "":
                continue
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def format_categories(categories) -> str:
    if not categories:
        return ""
    if isinstance(categories, list):
        if len(categories) > 0 and isinstance(categories[0], list):
            flat = categories[0]
        else:
            flat = categories
        return " > ".join(str(x) for x in flat if x)
    return clean_text(categories)


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
    elif isinstance(images, list):
        for image_record in images:
            if not isinstance(image_record, dict):
                continue
            for key in ("hi_res", "large", "thumb", "large_image_url", "medium_image_url", "small_image_url"):
                value = image_record.get(key)
                if isinstance(value, str) and value.strip() and value.strip() != "None":
                    candidates.append(value.strip())

    return candidates[0] if candidates else ""


def load_review_interactions(reviews_path: Path) -> List[Tuple[str, str, int]]:
    interactions: List[Tuple[str, str, int]] = []
    for record in iter_json_records(reviews_path, desc="Reading reviews"):
        user_id = extract_user_id(record)
        item_id = extract_item_id(record)
        if not user_id or not item_id:
            continue
        timestamp = extract_timestamp(record)
        interactions.append((user_id, item_id, timestamp))
    return interactions


def apply_iterative_k_core(
    interactions: Sequence[Tuple[str, str, int]],
    min_count: int = K_CORE,
) -> List[Tuple[str, str, int]]:
    filtered = list(interactions)
    iteration = 0
    while True:
        iteration += 1
        user_counter = Counter(user_id for user_id, _, _ in filtered)
        item_counter = Counter(item_id for _, item_id, _ in filtered)
        next_filtered = [
            (user_id, item_id, timestamp)
            for user_id, item_id, timestamp in filtered
            if user_counter[user_id] >= min_count and item_counter[item_id] >= min_count
        ]

        print(
            f"[5-core] iteration={iteration} "
            f"interactions={len(filtered)} -> {len(next_filtered)} "
            f"users={len(user_counter)} items={len(item_counter)}"
        )
        if len(next_filtered) == len(filtered):
            break
        filtered = next_filtered

    return filtered


def build_contiguous_mappings(
    interactions: Sequence[Tuple[str, str, int]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    user_ids = sorted({user_id for user_id, _, _ in interactions})
    item_ids = sorted({item_id for _, item_id, _ in interactions})
    user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    return user_map, item_map


def build_user_sequences(
    interactions: Sequence[Tuple[str, str, int]],
    user_map: Dict[str, int],
) -> Dict[int, List[str]]:
    grouped: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
    for user_id, item_id, timestamp in interactions:
        grouped[user_map[user_id]].append((timestamp, item_id))

    sequences: Dict[int, List[str]] = {}
    for remapped_user_id, events in tqdm(
        grouped.items(),
        total=len(grouped),
        desc="Sorting user sequences",
        unit="user",
    ):
        events.sort(key=lambda pair: (pair[0], pair[1]))
        sequences[remapped_user_id] = [item_id for _, item_id in events]
    return sequences


def select_item_universe_sequences(
    sequences: Dict[int, List[str]],
    *,
    item_universe_split: str,
    split_mode: str,
) -> Dict[int, List[str]]:
    if item_universe_split == "all" or split_mode == "none":
        return {user_id: list(sequence) for user_id, sequence in sequences.items()}

    selected: Dict[int, List[str]] = {}
    for user_id, sequence in sequences.items():
        if split_mode == "leave_last_two":
            selected[user_id] = list(sequence[:-2]) if len(sequence) >= 3 else []
        else:
            raise ValueError(f"Unsupported split_mode={split_mode!r}.")
    return selected


def build_item_map_from_sequences(sequences: Dict[int, List[str]]) -> Dict[str, int]:
    item_ids = sorted({item_id for sequence in sequences.values() for item_id in sequence})
    return {item_id: idx for idx, item_id in enumerate(item_ids)}


def load_item_meta(meta_path: Path, target_items: Sequence[str]) -> Dict[str, dict]:
    target_set = set(target_items)
    meta_dict: Dict[str, dict] = {}
    for record in iter_json_records(meta_path, desc="Reading metadata"):
        item_id = extract_item_id(record)
        if not item_id or item_id not in target_set:
            continue
        meta_dict[item_id] = {
            "title": clean_text(record.get("title", "")),
            "categories": format_categories(record.get("categories")),
            "image_url": choose_image_url(record),
        }
        if len(meta_dict) == len(target_set):
            break
    return meta_dict


def load_item_embeddings(item_map: Dict[str, int], embedding_path: Path, seed: int = 0) -> np.ndarray:
    """
    Load item embeddings with shape [num_items, 128].

    If the embedding file does not exist, return a deterministic dummy matrix
    generated from N(0, 1) for pipeline testing.
    """
    num_items = len(item_map)
    if embedding_path.exists():
        embeddings = np.load(embedding_path)
        if embeddings.shape != (num_items, EMBED_DIM):
            raise ValueError(
                "Embedding matrix shape mismatch. "
                f"Expected {(num_items, EMBED_DIM)}, got {embeddings.shape}."
            )
        return embeddings.astype(np.float32, copy=False)

    print(
        f"[warn] Embedding file not found at {embedding_path}. "
        "Generating dummy item embeddings with np.random.randn for testing."
    )
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_items, EMBED_DIM), dtype=np.float32)


class ImageStore:
    def __init__(
        self,
        image_root: Path,
        meta_dict: Dict[str, dict],
        *,
        image_size: int = IMAGE_SIZE,
        cache_size: int = IMAGE_CACHE_SIZE,
        download_timeout: float = 10.0,
        enable_download: bool = True,
    ) -> None:
        self.image_root = image_root
        self.meta_dict = meta_dict
        self.image_size = int(image_size)
        self.cache_size = int(cache_size)
        self.download_timeout = float(download_timeout)
        self.enable_download = bool(enable_download)
        self.zero_image = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._download_failures: set[str] = set()
        self.image_root.mkdir(parents=True, exist_ok=True)

    def _local_image_path(self, item_id: str) -> Path:
        return self.image_root / f"{item_id}.jpg"

    def _download_one(self, item_id: str) -> bool:
        local_path = self._local_image_path(item_id)
        if local_path.exists():
            return True
        if not self.enable_download or item_id in self._download_failures:
            return False

        meta = self.meta_dict.get(item_id, {})
        image_url = clean_text(meta.get("image_url", ""))
        if not image_url:
            self._download_failures.add(item_id)
            return False

        request = urllib.request.Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RecSys-DiT-Preprocess/1.0)"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.download_timeout) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert("RGB")
                image.save(local_path, format="JPEG", quality=95)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            self._download_failures.add(item_id)
            return False

    def prefetch_missing_images(self, item_ids: Sequence[str], num_workers: int = 16) -> None:
        if not self.enable_download:
            return

        pending = []
        for item_id in item_ids:
            local_path = self._local_image_path(item_id)
            meta = self.meta_dict.get(item_id, {})
            if local_path.exists():
                continue
            if clean_text(meta.get("image_url", "")):
                pending.append(item_id)

        if not pending:
            print("[image] No missing images require downloading.")
            return

        print(f"[image] Prefetching {len(pending)} missing images with {num_workers} threads.")
        with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
            futures = {executor.submit(self._download_one, item_id): item_id for item_id in pending}
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Downloading images", unit="img"):
                pass

    def process_image(self, item_id: str, meta_dict: Optional[dict] = None) -> np.ndarray:
        if item_id in self._cache:
            self._cache.move_to_end(item_id)
            return self._cache[item_id]

        if meta_dict is not None and item_id not in self.meta_dict:
            self.meta_dict[item_id] = meta_dict

        local_path = self._local_image_path(item_id)
        if not local_path.exists():
            self._download_one(item_id)

        if not local_path.exists():
            return self.zero_image

        try:
            with Image.open(local_path) as image:
                image = image.convert("RGB")
                image = ImageOps.fit(
                    image,
                    (self.image_size, self.image_size),
                    method=RESAMPLE_BICUBIC,
                    centering=(0.5, 0.5),
                )
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = np.transpose(array, (2, 0, 1))
        except (OSError, ValueError):
            array = self.zero_image

        self._cache[item_id] = array
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return array


def process_image(item_id: str, meta_dict: Optional[dict] = None) -> np.ndarray:
    """
    Load and preprocess a single item image.

    Priority:
    1. data/images/{item_id}.jpg
    2. Metadata URL download fallback
    3. All-zero image when the image is unavailable or corrupted
    """
    if IMAGE_STORE is None:
        raise RuntimeError("IMAGE_STORE is not initialized. Create ImageStore before calling process_image().")
    return IMAGE_STORE.process_image(item_id, meta_dict)


def target_text(item_id: str, meta_dict: Dict[str, dict]) -> str:
    meta = meta_dict.get(item_id, {})
    title = clean_text(meta.get("title", ""))
    categories = clean_text(meta.get("categories", ""))
    return title or categories or ""


def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    lock.acquire_write_lock()
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    finally:
        lock.release_lock()


def safe_write_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    lock.acquire_write_lock()
    try:
        with open(path, "wb") as handle:
            np.savez_compressed(handle, **arrays)
    finally:
        lock.release_lock()


def write_dirty_bit(chunk_dir: Path, chunk_size: int) -> None:
    dirty_bit = np.zeros(chunk_size, dtype=np.uint8)
    path = chunk_dir / "dirty_bit"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    lock.acquire_write_lock()
    try:
        with open(path, "wb") as handle:
            handle.write(dirty_bit.tobytes())
    finally:
        lock.release_lock()


def estimate_light_sample_bytes(history_len: int) -> int:
    history_item_ids_bytes = history_len * 4
    history_mask_bytes = history_len * 1
    target_item_id_bytes = 4
    user_id_bytes = 4
    return history_item_ids_bytes + history_mask_bytes + target_item_id_bytes + user_id_bytes


def build_lightweight_sample(
    *,
    user_id: int,
    sequence: Sequence[str],
    target_pos: int,
    item_map: Dict[str, int],
    history_len: int,
) -> Dict[str, np.ndarray]:
    target_item = sequence[target_pos]
    history_items = sequence[max(0, target_pos - history_len):target_pos]
    left_pad = history_len - len(history_items)

    history_item_ids = np.full((history_len,), -1, dtype=np.int32)
    history_mask = np.zeros((history_len,), dtype=np.uint8)

    for offset, item_id in enumerate(history_items):
        if item_id not in item_map:
            continue
        dst = left_pad + offset
        history_item_ids[dst] = item_map[item_id]
        history_mask[dst] = 1

    return {
        "user_id": np.int32(user_id),
        "history_item_ids": history_item_ids,
        "history_mask": history_mask,
        "target_item_id": np.int32(item_map[target_item]),
    }


def prepare_output_dir(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. "
                "Use --overwrite to delete it first."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def save_mapping_files(output_root: Path, user_map: Dict[str, int], item_map: Dict[str, int]) -> None:
    safe_write_json(output_root / "user_map.json", user_map)
    safe_write_json(output_root / "item_map.json", item_map)


def save_global_item_store(
    output_root: Path,
    item_embeddings: np.ndarray,
    item_meta: Dict[str, dict],
) -> None:
    np.save(output_root / "item_embeddings.npy", item_embeddings.astype(np.float32, copy=False))
    safe_write_json(output_root / "item_meta.json", item_meta)


def get_target_split(sequence_len: int, target_pos: int, split_mode: str) -> str:
    if split_mode == "none":
        return "train"
    if sequence_len < 2:
        raise ValueError(f"Expected sequence_len >= 2, but got {sequence_len}.")
    if target_pos == sequence_len - 1:
        return "test"
    if target_pos == sequence_len - 2:
        return "val"
    return "train"


def count_sequence_samples_by_split(
    sequence: Sequence[str],
    split_mode: str,
    *,
    known_items: Optional[set[str]] = None,
) -> Dict[str, int]:
    split_names = ["train"] if split_mode == "none" else list(SPLIT_NAMES)
    counts = {split_name: 0 for split_name in split_names}
    sequence_len = len(sequence)
    for target_pos in range(1, sequence_len):
        if known_items is not None and sequence[target_pos] not in known_items:
            continue
        split_name = get_target_split(sequence_len, target_pos, split_mode)
        counts[split_name] += 1
    return counts


def aggregate_split_counts(
    sequences: Dict[int, List[str]],
    split_mode: str,
    *,
    known_items: Optional[set[str]] = None,
) -> Dict[str, int]:
    split_names = ["train"] if split_mode == "none" else list(SPLIT_NAMES)
    counts = {split: 0 for split in split_names}
    for sequence in sequences.values():
        for split_name, count in count_sequence_samples_by_split(
            sequence,
            split_mode,
            known_items=known_items,
        ).items():
            counts[split_name] += count
    return counts


def build_split_write_plan(
    raw_count: int,
    *,
    chunk_size: int,
    max_samples: int,
) -> Dict[str, int]:
    samples_to_write = raw_count
    if max_samples > 0:
        samples_to_write = min(samples_to_write, max_samples)
    full_chunks = samples_to_write // chunk_size
    samples_to_write = full_chunks * chunk_size
    dropped_tail = raw_count - samples_to_write
    return {
        "raw_count": raw_count,
        "samples_to_write": samples_to_write,
        "full_chunks": full_chunks,
        "dropped_tail": dropped_tail,
    }


def build_stats_payload(
    *,
    args: argparse.Namespace,
    user_map: Dict[str, int],
    item_map: Dict[str, int],
    sequences: Dict[int, List[str]],
    samples_to_write: int,
    dropped_tail: int,
    num_chunks_written: int,
    total_samples_before_chunk_drop: int,
    split_name: Optional[str] = None,
) -> Dict[str, object]:
    payload = {
        "dataset_name": DATASET_NAME,
        "history_len": args.history_len,
        "chunk_size": args.chunk_size,
        "k_core": K_CORE,
        "item_universe_split": args.item_universe_split,
        "num_users": len(user_map),
        "num_items": len(item_map),
        "num_sequences": len(sequences),
        "total_samples_before_chunk_drop": total_samples_before_chunk_drop,
        "total_samples_written": samples_to_write,
        "dropped_tail_samples": dropped_tail,
        "num_chunks_written": num_chunks_written,
        "embedding_path": str(args.embedding_path),
        "reviews_path": str(args.reviews_path),
        "meta_path": str(args.meta_path),
        "image_root": str(args.image_root),
        "stats_only": bool(args.stats_only),
        "max_samples": int(args.max_samples),
        "storage_mode": "lightweight_chunk_index",
        "prefetch_images": bool(args.prefetch_images),
        "split_mode": args.split_mode,
    }
    if split_name is not None:
        payload["split_name"] = split_name
    return payload


def flush_chunk_samples(
    chunk_dir: Path,
    chunk_samples: List[Dict[str, np.ndarray]],
    chunk_size: int,
    history_len: int,
) -> None:
    if len(chunk_samples) != chunk_size:
        raise ValueError(
            f"Expected exactly {chunk_size} samples in a full chunk, got {len(chunk_samples)}."
        )

    payload = {
        "user_ids": np.zeros((chunk_size,), dtype=np.int32),
        "history_item_ids": np.full((chunk_size, history_len), -1, dtype=np.int32),
        "history_mask": np.zeros((chunk_size, history_len), dtype=np.uint8),
        "target_item_ids": np.zeros((chunk_size,), dtype=np.int32),
    }
    for idx, sample in enumerate(chunk_samples):
        payload["user_ids"][idx] = sample["user_id"]
        payload["history_item_ids"][idx] = sample["history_item_ids"]
        payload["history_mask"][idx] = sample["history_mask"]
        payload["target_item_ids"][idx] = sample["target_item_id"]

    write_dirty_bit(chunk_dir, chunk_size)
    safe_write_npz(chunk_dir / "samples.npz", payload)


def main() -> None:
    args = parse_args()

    if args.history_len < 1:
        raise ValueError("--history-len must be at least 1.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1.")

    if not args.reviews_path.exists():
        raise FileNotFoundError(f"Review file not found: {args.reviews_path}")
    if not args.meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {args.meta_path}")

    print("========== Amazon Preprocess ==========")
    print(f"reviews_path : {args.reviews_path}")
    print(f"meta_path    : {args.meta_path}")
    print(f"image_root   : {args.image_root}")
    print(f"output_root  : {args.output_root}")
    print(f"history_len  : {args.history_len}")
    print(f"chunk_size   : {args.chunk_size}")
    print(f"item_universe: {args.item_universe_split}")

    interactions = load_review_interactions(args.reviews_path)
    if not interactions:
        raise RuntimeError("No valid interactions were found in the review file.")
    print(f"[reviews] raw interactions: {len(interactions)}")

    filtered = apply_iterative_k_core(interactions, min_count=K_CORE)
    if not filtered:
        raise RuntimeError("5-core filtering removed all interactions.")
    del interactions
    gc.collect()

    user_map, _ = build_contiguous_mappings(filtered)

    sequences = build_user_sequences(filtered, user_map)
    del filtered
    gc.collect()

    universe_sequences = select_item_universe_sequences(
        sequences,
        item_universe_split=args.item_universe_split,
        split_mode=args.split_mode,
    )
    item_map = build_item_map_from_sequences(universe_sequences)
    if not item_map:
        raise RuntimeError(
            "The selected item universe is empty. "
            "Try using --item-universe-split=all or check the current split settings."
        )
    print(
        f"[mapping] users={len(user_map)} items={len(item_map)} "
        f"item_universe_split={args.item_universe_split}"
    )

    split_names = ["train"] if args.split_mode == "none" else list(SPLIT_NAMES)
    known_items = set(item_map.keys())
    split_raw_counts_all = aggregate_split_counts(sequences, args.split_mode)
    split_raw_counts = aggregate_split_counts(
        sequences,
        args.split_mode,
        known_items=known_items,
    )
    split_write_plans = {
        split_name: build_split_write_plan(
            split_raw_counts[split_name],
            chunk_size=args.chunk_size,
            max_samples=args.max_samples,
        )
        for split_name in split_names
    }

    total_samples = sum(plan["raw_count"] for plan in split_write_plans.values())
    total_samples_to_write = sum(plan["samples_to_write"] for plan in split_write_plans.values())
    total_dropped_tail = sum(plan["dropped_tail"] for plan in split_write_plans.values())

    if total_samples_to_write == 0:
        raise RuntimeError(
            "Not enough samples to form a full chunk under the current split settings. "
            f"Need at least {args.chunk_size} samples in at least one split."
        )

    light_sample_bytes = estimate_light_sample_bytes(args.history_len)
    approx_gib = (total_samples_to_write * light_sample_bytes) / (1024 ** 3)
    print(
        f"[samples] total={total_samples} write={total_samples_to_write} "
        f"drop_tail={total_dropped_tail} split_mode={args.split_mode}"
    )
    for split_name in split_names:
        plan = split_write_plans[split_name]
        dropped_unknown_targets = split_raw_counts_all[split_name] - split_raw_counts[split_name]
        print(
            f"[split:{split_name}] total={plan['raw_count']} "
            f"write={plan['samples_to_write']} "
            f"drop_tail={plan['dropped_tail']} "
            f"chunks={plan['full_chunks']} "
            f"dropped_unknown_targets={dropped_unknown_targets}"
        )
    print(f"[disk]    approx lightweight index payload={approx_gib:.4f} GiB")

    if args.split_mode == "none":
        stats = build_stats_payload(
            args=args,
            user_map=user_map,
            item_map=item_map,
            sequences=sequences,
            samples_to_write=split_write_plans["train"]["samples_to_write"],
            dropped_tail=split_write_plans["train"]["dropped_tail"],
            num_chunks_written=split_write_plans["train"]["full_chunks"],
            total_samples_before_chunk_drop=split_write_plans["train"]["raw_count"],
            split_name="train",
        )
        if args.stats_only:
            print("========== Stats Only ==========")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            return
    else:
        split_manifest = {
            "dataset_name": DATASET_NAME,
            "split_mode": args.split_mode,
            "history_len": args.history_len,
            "chunk_size": args.chunk_size,
            "k_core": K_CORE,
            "num_users": len(user_map),
            "num_items": len(item_map),
            "num_sequences": len(sequences),
            "embedding_path": str(args.embedding_path),
            "reviews_path": str(args.reviews_path),
            "meta_path": str(args.meta_path),
            "image_root": str(args.image_root),
            "max_samples": int(args.max_samples),
            "storage_mode": "lightweight_chunk_index",
            "splits": {
                split_name: {
                    "total_samples_before_chunk_drop": split_write_plans[split_name]["raw_count"],
                    "total_samples_written": split_write_plans[split_name]["samples_to_write"],
                    "dropped_tail_samples": split_write_plans[split_name]["dropped_tail"],
                    "num_chunks_written": split_write_plans[split_name]["full_chunks"],
                }
                for split_name in split_names
            },
        }
        if args.stats_only:
            print("========== Stats Only ==========")
            print(json.dumps(split_manifest, indent=2, ensure_ascii=False))
            return

    item_embeddings = load_item_embeddings(item_map, args.embedding_path, seed=args.seed)
    meta_dict = load_item_meta(args.meta_path, list(item_map.keys()))

    global IMAGE_STORE
    IMAGE_STORE = ImageStore(
        args.image_root,
        meta_dict,
        image_size=IMAGE_SIZE,
        cache_size=IMAGE_CACHE_SIZE,
        download_timeout=args.download_timeout,
        enable_download=not args.skip_download,
    )
    if args.prefetch_images:
        IMAGE_STORE.prefetch_missing_images(list(item_map.keys()), num_workers=args.download_workers)

    prepare_output_dir(args.output_root, overwrite=args.overwrite)
    split_roots = {
        split_name: (args.output_root if args.split_mode == "none" else args.output_root / split_name)
        for split_name in split_names
    }
    for split_name, split_root in split_roots.items():
        split_root.mkdir(parents=True, exist_ok=True)
        save_mapping_files(split_root, user_map, item_map)
        save_global_item_store(split_root, item_embeddings, meta_dict)

    if args.split_mode != "none":
        safe_write_json(args.output_root / "split_manifest.json", split_manifest)

    written_by_split = {split_name: 0 for split_name in split_names}
    current_chunk_samples: Dict[str, List[Dict[str, np.ndarray]]] = {
        split_name: [] for split_name in split_names
    }
    chunk_idx_by_split = {split_name: 0 for split_name in split_names}
    write_bar = tqdm(
        total=total_samples_to_write,
        desc="Writing chunk samples",
        unit="sample",
    )

    for remapped_user_id in sorted(sequences.keys()):
        sequence = sequences[remapped_user_id]
        for target_pos in range(1, len(sequence)):
            split_name = get_target_split(len(sequence), target_pos, args.split_mode)
            split_plan = split_write_plans[split_name]
            if written_by_split[split_name] >= split_plan["samples_to_write"]:
                continue
            if sequence[target_pos] not in item_map:
                continue

            sample_record = build_lightweight_sample(
                user_id=remapped_user_id,
                sequence=sequence,
                target_pos=target_pos,
                item_map=item_map,
                history_len=args.history_len,
            )
            current_chunk_samples[split_name].append(sample_record)
            written_by_split[split_name] += 1
            write_bar.update(1)

            if len(current_chunk_samples[split_name]) == args.chunk_size:
                chunk_dir = split_roots[split_name] / f"chunk_{chunk_idx_by_split[split_name]}"
                chunk_dir.mkdir(parents=True, exist_ok=False)
                flush_chunk_samples(
                    chunk_dir=chunk_dir,
                    chunk_samples=current_chunk_samples[split_name],
                    chunk_size=args.chunk_size,
                    history_len=args.history_len,
                )
                current_chunk_samples[split_name] = []
                chunk_idx_by_split[split_name] += 1

        if all(
            written_by_split[name] >= split_write_plans[name]["samples_to_write"]
            for name in split_names
        ):
            break

        if (remapped_user_id + 1) % 100 == 0:
            gc.collect()

    write_bar.close()

    for split_name in split_names:
        if current_chunk_samples[split_name]:
            raise RuntimeError(
                "Encountered a partially filled chunk at the end of writing. "
                f"This should not happen after dropping tail samples for split `{split_name}`."
            )

        split_stats = build_stats_payload(
            args=args,
            user_map=user_map,
            item_map=item_map,
            sequences=sequences,
            samples_to_write=written_by_split[split_name],
            dropped_tail=split_write_plans[split_name]["dropped_tail"],
            num_chunks_written=chunk_idx_by_split[split_name],
            total_samples_before_chunk_drop=split_write_plans[split_name]["raw_count"],
            split_name=split_name,
        )
        safe_write_json(split_roots[split_name] / "stats.json", split_stats)

    print("========== Done ==========")
    if args.split_mode == "none":
        final_payload = build_stats_payload(
            args=args,
            user_map=user_map,
            item_map=item_map,
            sequences=sequences,
            samples_to_write=written_by_split["train"],
            dropped_tail=split_write_plans["train"]["dropped_tail"],
            num_chunks_written=chunk_idx_by_split["train"],
            total_samples_before_chunk_drop=split_write_plans["train"]["raw_count"],
            split_name="train",
        )
    else:
        final_payload = {
            "dataset_name": DATASET_NAME,
            "split_mode": args.split_mode,
            "output_root": str(args.output_root),
            "splits": {
                split_name: build_stats_payload(
                    args=args,
                    user_map=user_map,
                    item_map=item_map,
                    sequences=sequences,
                    samples_to_write=written_by_split[split_name],
                    dropped_tail=split_write_plans[split_name]["dropped_tail"],
                    num_chunks_written=chunk_idx_by_split[split_name],
                    total_samples_before_chunk_drop=split_write_plans[split_name]["raw_count"],
                    split_name=split_name,
                )
                for split_name in split_names
            },
        }
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
