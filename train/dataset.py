import ast
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from data.filelock import FileLock
from train.image_corrupt import image_corrupt


def get_clean_item(chunk_dir):
    """
    Get indexes of clean items in a chunk.
    """
    dirty_bit = read_dirty_bit(chunk_dir)
    return np.where(1 - dirty_bit)[0].tolist()


def save_dirty_bit(chunk_dir, dirty_bit):
    """
    Save the dirty bit to the chunk directory.
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        lock = None
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_write_lock()
            with open(file_path, "wb") as file:
                file.write(dirty_bit.tobytes())
            lock.release_lock()
            return
        except KeyboardInterrupt:
            if lock is not None:
                lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            if lock is not None:
                lock.release_lock()
            continue
    raise RuntimeError("Failed to save dirty bit.")


def read_dirty_bit(chunk_dir):
    """
    Read the dirty bit from the chunk directory.
    """
    time_stmp = time.time()
    while time.time() - time_stmp < 10.0:
        lock = None
        try:
            file_path = os.path.join(chunk_dir, "dirty_bit")
            lock = FileLock(file_path)
            lock.acquire_read_lock()
            with open(file_path, "rb") as file:
                dirty_bit = np.frombuffer(file.read(), dtype=np.uint8).copy()
            lock.release_lock()
            assert len(dirty_bit) > 0
            return dirty_bit
        except KeyboardInterrupt:
            if lock is not None:
                lock.release_lock()
            raise KeyboardInterrupt
        except BaseException:
            if lock is not None:
                lock.release_lock()
            continue
    raise RuntimeError("Failed to read dirty bit.")


class VLAConsumerDataset(Dataset):
    """Sequence recommendation dataset with multimodal history conditions."""

    DATASET_NAME = "sequence_recommendation"
    EMBED_DIM = 128
    IMAGE_SIZE = 224

    def __init__(
        self,
        config,
        image_processor,
        img_history_size,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        cond_mask_prob=0.1,
    ):
        super(VLAConsumerDataset, self).__init__()

        self.config = config
        self.image_processor = image_processor
        self.image_size = int(image_size or self.IMAGE_SIZE)
        self.history_len = max(1, int(config.get("history_len", img_history_size)))
        self.cond_mask_prob = float(cond_mask_prob)
        self.text_branch_mask_prob = float(
            config.get("text_branch_mask_prob", self.cond_mask_prob)
        )
        self.image_branch_mask_prob = float(
            config.get("image_branch_mask_prob", self.cond_mask_prob)
        )
        self.image_aug = bool(image_aug)
        self.auto_adjust_image_brightness = bool(auto_adjust_image_brightness)
        self.max_review_chars = max(32, int(config.get("max_review_chars", 160)))
        self.max_text_history = max(1, int(config.get("max_text_history", self.history_len)))
        self.min_history_len = max(1, int(config.get("min_history_len", 1)))
        self.max_samples = int(config.get("max_samples", 0)) or None
        self.ctrl_freq = int(config.get("ctrl_freq", 1))
        self.dataset_name2id = {self.DATASET_NAME: 0}
        self.dataset_id2name = {0: self.DATASET_NAME}

        self.zero_embed = torch.zeros(self.EMBED_DIM, dtype=torch.float32)
        self.zero_image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
        self.image_mean = list(getattr(self.image_processor, "image_mean", [0.485, 0.456, 0.406]))
        self.image_std = list(getattr(self.image_processor, "image_std", [0.229, 0.224, 0.225]))
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

        self.image_root = None
        self.item_meta: Dict[str, dict] = {}
        self.image_path_cache: Dict[str, Optional[Path]] = {}
        self.item_embed_cache: Dict[object, torch.Tensor] = {}
        self.sample_refs: List[Tuple[int, int]] = []

        self.buffer_root = self._resolve_optional_path(
            config.get("buffer_root") or config.get("preprocessed_buffer_root")
        )
        self.buffer_mode = bool(
            self.buffer_root is not None and (self.buffer_root / "stats.json").exists()
        )

        self.sequences: List[List[dict]] = []
        self.reviews_path = None
        self.meta_path = None
        self.item_embedding_lookup = None

        self.stats = {}
        self.chunk_dirs: List[Path] = []
        self.idx_to_item: Dict[int, str] = {}
        self.item_embeddings = None
        self._cached_chunk_idx: Optional[int] = None
        self._cached_chunk_arrays: Optional[Dict[str, np.ndarray]] = None

        if self.buffer_mode:
            self._init_buffer_mode()
        else:
            self._init_raw_mode()

        if len(self.sample_refs) == 0:
            raise RuntimeError("No valid recommendation samples were built.")

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    def __len__(self) -> int:
        return len(self.sample_refs)

    @staticmethod
    def _clean_text(value):
        if value is None:
            return ""
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        return " ".join(str(value).split())

    @staticmethod
    def _read_line_record(line):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return ast.literal_eval(line)

    @staticmethod
    def _format_categories(categories):
        if not categories:
            return ""
        if isinstance(categories, list):
            first = categories[0]
            if isinstance(first, list):
                return " > ".join(str(x) for x in first if x)
            return " > ".join(str(x) for x in categories if x)
        return str(categories)

    def _truncate_text(self, value):
        value = self._clean_text(value)
        if len(value) <= self.max_review_chars:
            return value
        return value[: self.max_review_chars].rstrip() + "..."

    def _resolve_optional_path(self, path_like):
        if not path_like:
            return None
        path = Path(path_like)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _resolve_required_path(self, explicit_path, fallback_candidates):
        candidates = []
        if explicit_path:
            candidates.append(explicit_path)
        candidates.extend(fallback_candidates)
        for candidate in candidates:
            path = Path(candidate)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not find dataset file. Checked: {', '.join(str(Path(c)) for c in candidates)}"
        )

    def _open_text(self, file_path: Path):
        if file_path.suffix.lower() == ".gz":
            return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
        return open(file_path, "r", encoding="utf-8", errors="replace")

    def _iter_records(self, file_path):
        with self._open_text(file_path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                yield self._read_line_record(line)

    def _load_json(self, path: Path):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _load_item_embedding_lookup(self, path_like):
        path = self._resolve_optional_path(path_like)
        if path is None or not path.exists():
            return None

        suffix = path.suffix.lower()
        if suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu")
        elif suffix == ".json":
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        elif suffix == ".npz":
            payload = dict(np.load(path, allow_pickle=True))
        else:
            raise ValueError(f"Unsupported item embedding file format: {path}")

        if isinstance(payload, dict):
            mapping_keys = ["item_to_idx", "id_to_idx", "asin_to_idx"]
            embed_keys = ["embeddings", "item_embeds", "id_embeds"]
            mapping = next((payload[k] for k in mapping_keys if k in payload), None)
            embeddings = next((payload[k] for k in embed_keys if k in payload), None)
            if mapping is not None and embeddings is not None:
                return {"type": "indexed", "mapping": dict(mapping), "embeddings": embeddings}
            return {"type": "direct", "mapping": payload}

        raise ValueError(f"Unsupported item embedding payload type: {type(payload).__name__}")

    def _coerce_embedding(self, value):
        tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        tensor = tensor.detach().cpu().to(torch.float32).reshape(-1)
        if tensor.numel() != self.EMBED_DIM:
            raise ValueError(
                f"Expected item embedding dim {self.EMBED_DIM}, got shape {tuple(tensor.shape)}"
            )
        return tensor

    def _hashed_embedding(self, asin):
        seed = int(hashlib.sha256(asin.encode("utf-8")).hexdigest()[:16], 16)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        embedding = torch.randn(self.EMBED_DIM, generator=generator, dtype=torch.float32)
        return embedding / embedding.norm(p=2).clamp_min(1e-6)

    def _get_item_embedding(self, asin):
        if asin in self.item_embed_cache:
            return self.item_embed_cache[asin]

        embedding = None
        if self.item_embedding_lookup is not None:
            if self.item_embedding_lookup["type"] == "direct":
                raw_value = self.item_embedding_lookup["mapping"].get(asin)
                if raw_value is not None:
                    embedding = self._coerce_embedding(raw_value)
            else:
                item_idx = self.item_embedding_lookup["mapping"].get(asin)
                if item_idx is not None:
                    embedding = self._coerce_embedding(
                        self.item_embedding_lookup["embeddings"][item_idx]
                    )

        if embedding is None:
            embedding = self._hashed_embedding(asin)

        self.item_embed_cache[asin] = embedding
        return embedding

    def _get_buffer_embedding(self, item_idx: int) -> torch.Tensor:
        cache_key = ("buffer", int(item_idx))
        if cache_key in self.item_embed_cache:
            return self.item_embed_cache[cache_key]

        if self.item_embeddings is None:
            raise RuntimeError("Buffer embedding matrix is not initialized.")
        vector = np.asarray(self.item_embeddings[item_idx], dtype=np.float32)
        tensor = torch.from_numpy(vector.copy())
        self.item_embed_cache[cache_key] = tensor
        return tensor

    def _load_item_meta(self, item_ids):
        item_meta = {}
        for record in self._iter_records(self.meta_path):
            asin = str(record.get("asin", "")).strip()
            if not asin or asin not in item_ids:
                continue
            item_meta[asin] = {
                "title": self._clean_text(record.get("title", "")),
                "brand": self._clean_text(record.get("brand", "")),
                "categories": self._format_categories(record.get("categories")),
                "im_url": self._clean_text(record.get("imUrl", "")),
                "image_path": self._clean_text(record.get("image_path", "")),
            }
            if len(item_meta) == len(item_ids):
                break
        return item_meta

    def _init_raw_mode(self):
        self.reviews_path = self._resolve_required_path(
            self.config.get("reviews_path"),
            [
                "data/Amazon_Sport/reviews_Sports_and_Outdoors_5.json",
                "data/Amazon_Sport/reviews_Sports_and_Outdoors_5.jsonl",
                "data/Amazon_Music_And_Instruments/Digital_Music_5.json",
                "data/Amazon_Music_And_Instruments/Digital_Music_5.jsonl",
                "data/Amazon_Music_And_Instruments/Musical_Instruments_5.json",
                "data/Amazon_Music_And_Instruments/Musical_Instruments_5.json.gz",
            ],
        )
        self.meta_path = self._resolve_required_path(
            self.config.get("meta_path"),
            [
                "data/Amazon_Sport/meta_Sports_and_Outdoors.json",
                "data/Amazon_Sport/meta_Sports_and_Outdoors.jsonl",
                "data/Amazon_Music_And_Instruments/meta_Digital_Music.json",
                "data/Amazon_Music_And_Instruments/meta_Digital_Music.jsonl",
                "data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json",
                "data/Amazon_Music_And_Instruments/meta_Musical_Instruments.json.gz",
            ],
        )
        self.image_root = self._resolve_optional_path(self.config.get("image_root"))
        self.item_embedding_lookup = self._load_item_embedding_lookup(
            self.config.get("item_embed_path")
        )
        self._build_raw_dataset_index()

    def _init_buffer_mode(self):
        self.stats = self._load_json(self.buffer_root / "stats.json")
        stats_history_len = int(self.stats.get("history_len", self.history_len))
        if stats_history_len != self.history_len:
            raise ValueError(
                f"Config history_len={self.history_len} does not match buffer history_len={stats_history_len}."
            )

        self.item_meta = self._load_json(self.buffer_root / "item_meta.json")
        item_map = self._load_json(self.buffer_root / "item_map.json")
        self.idx_to_item = {int(idx): item_id for item_id, idx in item_map.items()}
        self.item_embeddings = np.load(
            self.buffer_root / "item_embeddings.npy",
            mmap_mode="r",
        )
        if self.item_embeddings.ndim != 2 or self.item_embeddings.shape[1] != self.EMBED_DIM:
            raise ValueError(
                "Expected `item_embeddings.npy` with shape [num_items, 128], "
                f"got {tuple(self.item_embeddings.shape)}."
            )
        if self.item_embeddings.shape[0] != len(self.idx_to_item):
            raise ValueError(
                "Mismatch between item embedding rows and item_map size: "
                f"{self.item_embeddings.shape[0]} vs {len(self.idx_to_item)}."
            )

        self.image_root = self._resolve_optional_path(
            self.config.get("image_root") or self.stats.get("image_root")
        )
        self.chunk_dirs = sorted(
            [
                path
                for path in self.buffer_root.glob("chunk_*")
                if path.is_dir()
                and (path / "samples.npz").exists()
                and (path / "dirty_bit").exists()
            ],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        if not self.chunk_dirs:
            raise RuntimeError(f"No chunk directories found under {self.buffer_root}.")

        self._build_buffer_index()

    def _build_raw_dataset_index(self):
        user_sequences = {}
        item_ids = set()

        for record in self._iter_records(self.reviews_path):
            reviewer_id = str(record.get("reviewerID", "")).strip()
            asin = str(record.get("asin", "")).strip()
            if not reviewer_id or not asin:
                continue

            interaction = {
                "asin": asin,
                "timestamp": int(record.get("unixReviewTime", 0) or 0),
                "summary": self._clean_text(record.get("summary", "")),
                "review_text": self._truncate_text(record.get("reviewText", "")),
            }
            user_sequences.setdefault(reviewer_id, []).append(interaction)
            item_ids.add(asin)

        self.item_meta = self._load_item_meta(item_ids)

        for sequence in user_sequences.values():
            sequence.sort(key=lambda x: (x["timestamp"], x["asin"]))
            if len(sequence) <= self.min_history_len:
                continue

            seq_idx = len(self.sequences)
            self.sequences.append(sequence)
            for target_pos in range(self.min_history_len, len(sequence)):
                self.sample_refs.append((seq_idx, target_pos))
                if self.max_samples is not None and len(self.sample_refs) >= self.max_samples:
                    break
            if self.max_samples is not None and len(self.sample_refs) >= self.max_samples:
                break

    def _build_buffer_index(self):
        for chunk_idx, chunk_dir in enumerate(self.chunk_dirs):
            clean_indices = get_clean_item(chunk_dir)
            for sample_idx in clean_indices:
                self.sample_refs.append((chunk_idx, sample_idx))
                if self.max_samples is not None and len(self.sample_refs) >= self.max_samples:
                    return

    def _load_chunk_arrays(self, chunk_idx: int) -> Dict[str, np.ndarray]:
        if self._cached_chunk_idx == chunk_idx and self._cached_chunk_arrays is not None:
            return self._cached_chunk_arrays

        chunk_dir = self.chunk_dirs[chunk_idx]
        samples_path = chunk_dir / "samples.npz"
        time_stmp = time.time()
        while time.time() - time_stmp < 10.0:
            lock = None
            try:
                lock = FileLock(str(samples_path))
                lock.acquire_read_lock()
                with np.load(samples_path, allow_pickle=False) as payload:
                    arrays = {key: payload[key].copy() for key in payload.files}
                lock.release_lock()
                self._cached_chunk_idx = chunk_idx
                self._cached_chunk_arrays = arrays
                return arrays
            except KeyboardInterrupt:
                if lock is not None:
                    lock.release_lock()
                raise KeyboardInterrupt
            except BaseException:
                if lock is not None:
                    lock.release_lock()
                continue
        raise RuntimeError(f"Failed to read chunk file: {samples_path}")

    def _resolve_image_path(self, asin):
        if asin in self.image_path_cache:
            return self.image_path_cache[asin]

        resolved_path = None
        if self.image_root is not None:
            meta = self.item_meta.get(asin, {})
            candidates = [
                self.image_root / f"{asin}.jpg",
                self.image_root / f"{asin}.jpeg",
                self.image_root / f"{asin}.png",
                self.image_root / f"{asin}.webp",
            ]
            if meta.get("image_path"):
                candidates.append(self.image_root / meta["image_path"])
            image_url = meta.get("im_url") or meta.get("image_url") or ""
            if image_url:
                filename = str(image_url).split("?")[0].rsplit("/", 1)[-1]
                if filename:
                    candidates.append(self.image_root / filename)
                    candidates.append(self.image_root / asin / filename)
            for candidate in candidates:
                if candidate.exists():
                    resolved_path = candidate
                    break

        self.image_path_cache[asin] = resolved_path
        return resolved_path

    def _load_image_tensor(self, asin):
        image_path = self._resolve_image_path(asin)
        if image_path is None:
            return self.zero_image.clone()

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            if self.auto_adjust_image_brightness:
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (
                    len(pixel_values) * 255.0 * 3
                )
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.image_aug and random.random() > 0.5:
                aug_type = random.choice(["corrput_only", "color_only", "both"])
                if aug_type != "corrput_only":
                    image = transforms.ColorJitter(
                        brightness=0.3,
                        contrast=0.4,
                        saturation=0.5,
                        hue=0.03,
                    )(image)
                if aug_type != "color_only":
                    image = image_corrupt(image)

            return self.image_transform(image)
        except BaseException:
            return self.zero_image.clone()

    def _target_text_from_asin(self, asin: str) -> str:
        meta = self.item_meta.get(asin, {})
        title = self._clean_text(meta.get("title", ""))
        categories = self._clean_text(meta.get("categories", ""))
        return title or categories or ""

    def _build_text(self, history_events):
        if len(history_events) == 0:
            return ""

        lines = ["Recommend the next item from this interaction history:"]
        for idx, event in enumerate(history_events[-self.max_text_history :], start=1):
            meta = self.item_meta.get(event["asin"], {})
            title = meta.get("title") or event["asin"]
            fields = [f"{idx}. title: {title}"]
            if meta.get("brand"):
                fields.append(f"brand: {meta['brand']}")
            if meta.get("categories"):
                fields.append(f"category: {meta['categories']}")
            feedback = event["summary"] or event["review_text"]
            if feedback:
                fields.append(f"feedback: {feedback}")
            lines.append("; ".join(fields))
        return "\n".join(lines)

    def _maybe_mask_conditions(
        self,
        history_id_embeds: torch.Tensor,
        history_pixel_values: torch.Tensor,
        target_pixel_values: torch.Tensor,
        text: str,
    ):
        full_cond_masked = random.random() < self.cond_mask_prob
        image_branch_masked = full_cond_masked or (
            random.random() < self.image_branch_mask_prob
        )
        text_branch_masked = full_cond_masked or (
            random.random() < self.text_branch_mask_prob
        )

        if image_branch_masked:
            history_id_embeds = torch.zeros_like(history_id_embeds)
            history_pixel_values = torch.zeros_like(history_pixel_values)
            target_pixel_values = torch.zeros_like(target_pixel_values)
        if text_branch_masked:
            text = ""

        all_branches_masked = image_branch_masked and text_branch_masked
        return (
            history_id_embeds,
            history_pixel_values,
            target_pixel_values,
            text,
            all_branches_masked,
        )

    def _get_raw_sample(self, index):
        seq_idx, target_pos = self.sample_refs[index]
        sequence = self.sequences[seq_idx]
        history_events = sequence[max(0, target_pos - self.history_len) : target_pos]
        target_event = sequence[target_pos]

        history_embeds = [self._get_item_embedding(event["asin"]) for event in history_events]
        history_images = [self._load_image_tensor(event["asin"]) for event in history_events]
        pad_len = self.history_len - len(history_embeds)
        if pad_len > 0:
            history_embeds = [self.zero_embed.clone() for _ in range(pad_len)] + history_embeds
            history_images = [self.zero_image.clone() for _ in range(pad_len)] + history_images

        history_id_embeds = torch.stack(history_embeds, dim=0)
        history_pixel_values = torch.stack(history_images, dim=0)
        target_pixel_values = self._load_image_tensor(target_event["asin"])
        target_embed = self._get_item_embedding(target_event["asin"]).unsqueeze(0)
        text = self._build_text(history_events)
        history_item_ids = torch.full((self.history_len,), -1, dtype=torch.long)
        history_mask = torch.zeros((self.history_len,), dtype=torch.bool)
        pad_len = self.history_len - len(history_events)
        if len(history_events) > 0:
            history_mask[pad_len:] = True
        target_item_id = torch.tensor(-1, dtype=torch.long)
        return (
            history_id_embeds,
            history_pixel_values,
            target_pixel_values,
            text,
            target_embed,
            history_item_ids,
            history_mask,
            target_item_id,
        )

    def _get_buffer_sample(self, index):
        chunk_idx, sample_idx = self.sample_refs[index]
        arrays = self._load_chunk_arrays(chunk_idx)

        history_item_ids = arrays["history_item_ids"][sample_idx]
        history_mask = arrays["history_mask"][sample_idx]
        target_item_id = int(arrays["target_item_ids"][sample_idx])
        target_asin = self.idx_to_item[target_item_id]

        history_embeds: List[torch.Tensor] = []
        history_images: List[torch.Tensor] = []
        for item_idx, mask in zip(history_item_ids.tolist(), history_mask.tolist()):
            if int(mask) == 0 or int(item_idx) < 0:
                history_embeds.append(self.zero_embed.clone())
                history_images.append(self.zero_image.clone())
                continue
            asin = self.idx_to_item[int(item_idx)]
            history_embeds.append(self._get_buffer_embedding(int(item_idx)))
            history_images.append(self._load_image_tensor(asin))

        history_id_embeds = torch.stack(history_embeds, dim=0)
        history_pixel_values = torch.stack(history_images, dim=0)
        target_pixel_values = self._load_image_tensor(target_asin)
        target_embed = self._get_buffer_embedding(target_item_id).unsqueeze(0)
        text = self._target_text_from_asin(target_asin)
        history_item_ids = torch.tensor(history_item_ids, dtype=torch.long)
        history_mask = torch.tensor(history_mask, dtype=torch.bool)
        target_item_id = torch.tensor(target_item_id, dtype=torch.long)
        return (
            history_id_embeds,
            history_pixel_values,
            target_pixel_values,
            text,
            target_embed,
            history_item_ids,
            history_mask,
            target_item_id,
        )

    def __getitem__(self, index):
        if self.buffer_mode:
            (
                history_id_embeds,
                history_pixel_values,
                target_pixel_values,
                text,
                target_embed,
                history_item_ids,
                history_mask,
                target_item_id,
            ) = self._get_buffer_sample(index)
        else:
            (
                history_id_embeds,
                history_pixel_values,
                target_pixel_values,
                text,
                target_embed,
                history_item_ids,
                history_mask,
                target_item_id,
            ) = self._get_raw_sample(index)

        (
            history_id_embeds,
            history_pixel_values,
            target_pixel_values,
            text,
            cond_masked,
        ) = self._maybe_mask_conditions(
            history_id_embeds=history_id_embeds,
            history_pixel_values=history_pixel_values,
            target_pixel_values=target_pixel_values,
            text=text,
        )

        return {
            "history_id_embeds": history_id_embeds,
            "history_pixel_values": history_pixel_values,
            "target_pixel_values": target_pixel_values,
            "text": text,
            "target_embed": target_embed,
            "target_item_id": target_item_id,
            "history_item_ids": history_item_ids,
            "history_mask": history_mask,
            "data_idx": self.dataset_name2id[self.DATASET_NAME],
            "ctrl_freq": 0 if cond_masked else self.ctrl_freq,
        }


class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "history_id_embeds": torch.stack(
                [instance["history_id_embeds"] for instance in instances], dim=0
            ),
            "history_pixel_values": torch.stack(
                [instance["history_pixel_values"] for instance in instances], dim=0
            ),
            "target_pixel_values": torch.stack(
                [instance["target_pixel_values"] for instance in instances], dim=0
            ),
            "target_embed": torch.stack(
                [instance["target_embed"] for instance in instances], dim=0
            ),
            "target_item_ids": torch.stack(
                [instance["target_item_id"] for instance in instances], dim=0
            ),
            "history_item_ids": torch.stack(
                [instance["history_item_ids"] for instance in instances], dim=0
            ),
            "history_masks": torch.stack(
                [instance["history_mask"] for instance in instances], dim=0
            ),
            "text": [instance["text"] for instance in instances],
            "data_indices": torch.tensor(
                [instance.get("data_idx", 0) for instance in instances], dtype=torch.long
            ),
            "ctrl_freqs": torch.tensor(
                [instance.get("ctrl_freq", 1) for instance in instances], dtype=torch.long
            ),
        }
