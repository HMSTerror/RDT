from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class GenRecTokenizedDataset(Dataset):
    """Tokenized semantic-ID dataset with optional multimodal condition branches."""

    def __init__(
        self,
        *,
        tokenized_root: str | Path,
        split: str,
        text_embedding_path: str | Path | None = None,
        image_embedding_path: str | Path | None = None,
        cf_embedding_path: str | Path | None = None,
        target_latent_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.tokenized_root = Path(tokenized_root)
        self.split = split

        manifest_path = self.tokenized_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Tokenized manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as fp:
            self.manifest = json.load(fp)

        split_root = self.tokenized_root / split
        if not split_root.exists():
            raise FileNotFoundError(f"Split directory not found: {split_root}")

        samples_path = split_root / "samples.npz"
        if not samples_path.exists():
            raise FileNotFoundError(f"Tokenized sample file not found: {samples_path}")

        with np.load(samples_path, allow_pickle=False) as payload:
            self.arrays = {key: payload[key].copy() for key in payload.files}

        required_array_keys = (
            "input_ids",
            "attention_mask",
            "labels",
            "token_slot_ids",
            "token_codebook_ids",
            "seq_lengths",
            "history_item_ids",
            "history_mask",
            "target_item_ids",
            "history_semantic_ids",
            "target_semantic_ids",
        )
        missing_keys = [key for key in required_array_keys if key not in self.arrays]
        if missing_keys:
            raise KeyError(
                f"Tokenized sample file {samples_path} is missing required arrays: {missing_keys}"
            )

        target_max = int(np.max(self.arrays["target_item_ids"])) if self.arrays["target_item_ids"].size else -1
        history_max = int(np.max(self.arrays["history_item_ids"])) if self.arrays["history_item_ids"].size else -1
        self.max_item_id = max(target_max, history_max)

        self.text_embeddings = self._load_optional_matrix(text_embedding_path, min_rows=self.max_item_id + 1)
        self.image_embeddings = self._load_optional_matrix(image_embedding_path, min_rows=self.max_item_id + 1)
        self.cf_embeddings = self._load_optional_matrix(cf_embedding_path, min_rows=self.max_item_id + 1)
        self.target_latents = self._load_optional_matrix(target_latent_path, min_rows=self.max_item_id + 1)
        self.zero_text = self._zero_vector(self.text_embeddings)
        self.zero_image = self._zero_vector(self.image_embeddings)
        self.zero_cf = self._zero_vector(self.cf_embeddings)
        self.zero_target_latent = self._zero_vector(self.target_latents)

    @staticmethod
    def _load_optional_matrix(
        path_like: str | Path | None,
        *,
        min_rows: int,
    ) -> Optional[np.ndarray]:
        if path_like is None:
            return None
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Condition embedding file not found: {path}")
        matrix = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D condition matrix, got shape {tuple(matrix.shape)}.")
        if matrix.shape[0] < min_rows:
            raise ValueError(
                f"Condition matrix {path} only has {matrix.shape[0]} rows, "
                f"but tokenized samples require at least {min_rows}."
            )
        return matrix

    @staticmethod
    def _zero_vector(matrix: Optional[np.ndarray]) -> Optional[torch.Tensor]:
        if matrix is None:
            return None
        return torch.zeros(matrix.shape[1], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.arrays["target_item_ids"].shape[0])

    def _gather_history_branch(
        self,
        matrix: Optional[np.ndarray],
        zero_vector: Optional[torch.Tensor],
        history_item_ids: np.ndarray,
        history_mask: np.ndarray,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if matrix is None or zero_vector is None:
            return None, None

        history_vectors = []
        valid_vectors = []
        for item_idx, is_valid in zip(history_item_ids.tolist(), history_mask.tolist()):
            if not bool(is_valid) or int(item_idx) < 0:
                history_vectors.append(zero_vector.clone())
                continue
            vector = torch.from_numpy(np.asarray(matrix[int(item_idx)], dtype=np.float32).copy())
            history_vectors.append(vector)
            valid_vectors.append(vector)

        history_tensor = torch.stack(history_vectors, dim=0)
        if valid_vectors:
            pooled = torch.stack(valid_vectors, dim=0).mean(dim=0)
        else:
            pooled = zero_vector.clone()
        return history_tensor, pooled

    def _gather_target_branch(
        self,
        matrix: Optional[np.ndarray],
        zero_vector: Optional[torch.Tensor],
        target_item_id: int,
    ) -> Optional[torch.Tensor]:
        if matrix is None or zero_vector is None:
            return None
        if target_item_id < 0:
            return zero_vector.clone()
        return torch.from_numpy(np.asarray(matrix[int(target_item_id)], dtype=np.float32).copy())

    def __getitem__(self, index: int) -> Dict:
        history_item_ids = self.arrays["history_item_ids"][index]
        history_mask = self.arrays["history_mask"][index]
        target_item_id = int(self.arrays["target_item_ids"][index])

        history_text_embeds, pooled_text_embed = self._gather_history_branch(
            self.text_embeddings,
            self.zero_text,
            history_item_ids,
            history_mask,
        )
        target_text_embed = self._gather_target_branch(
            self.text_embeddings,
            self.zero_text,
            target_item_id,
        )
        history_image_embeds, pooled_image_embed = self._gather_history_branch(
            self.image_embeddings,
            self.zero_image,
            history_item_ids,
            history_mask,
        )
        target_image_embed = self._gather_target_branch(
            self.image_embeddings,
            self.zero_image,
            target_item_id,
        )
        history_cf_embeds, pooled_cf_embed = self._gather_history_branch(
            self.cf_embeddings,
            self.zero_cf,
            history_item_ids,
            history_mask,
        )
        target_cf_embed = self._gather_target_branch(
            self.cf_embeddings,
            self.zero_cf,
            target_item_id,
        )

        sample = {
            "input_ids": torch.from_numpy(self.arrays["input_ids"][index].astype(np.int64, copy=False)),
            "attention_mask": torch.from_numpy(self.arrays["attention_mask"][index].astype(np.bool_, copy=False)),
            "labels": torch.from_numpy(self.arrays["labels"][index].astype(np.int64, copy=False)),
            "token_slot_ids": torch.from_numpy(self.arrays["token_slot_ids"][index].astype(np.int64, copy=False)),
            "token_codebook_ids": torch.from_numpy(self.arrays["token_codebook_ids"][index].astype(np.int64, copy=False)),
            "history_item_ids": torch.from_numpy(history_item_ids.astype(np.int64, copy=False)),
            "history_masks": torch.from_numpy(history_mask.astype(np.bool_, copy=False)),
            "target_item_id": torch.tensor(target_item_id, dtype=torch.long),
            "history_semantic_ids": torch.from_numpy(self.arrays["history_semantic_ids"][index].astype(np.int64, copy=False)),
            "target_semantic_ids": torch.from_numpy(self.arrays["target_semantic_ids"][index].astype(np.int64, copy=False)),
            "seq_length": torch.tensor(int(self.arrays["seq_lengths"][index]), dtype=torch.long),
        }

        if history_text_embeds is not None:
            sample["history_text_embeds"] = history_text_embeds
            sample["target_text_embed"] = target_text_embed
            sample["pooled_text_embed"] = pooled_text_embed
        if history_image_embeds is not None:
            sample["history_image_embeds"] = history_image_embeds
            sample["target_image_embed"] = target_image_embed
            sample["pooled_image_embed"] = pooled_image_embed
        if history_cf_embeds is not None:
            sample["history_cf_embeds"] = history_cf_embeds
            sample["target_cf_embed"] = target_cf_embed
            sample["pooled_cf_embed"] = pooled_cf_embed
        if self.target_latents is not None:
            sample["target_item_latent"] = self._gather_target_branch(
                self.target_latents,
                self.zero_target_latent,
                target_item_id,
            )
        return sample


class GenRecTokenizedCollator:
    """Collate tokenized semantic-ID samples with optional condition branches."""

    OPTIONAL_STACK_KEYS = (
        "history_text_embeds",
        "target_text_embed",
        "pooled_text_embed",
        "history_image_embeds",
        "target_image_embed",
        "pooled_image_embed",
        "history_cf_embeds",
        "target_cf_embed",
        "pooled_cf_embed",
        "target_item_latent",
    )

    REQUIRED_STACK_KEYS = (
        "input_ids",
        "attention_mask",
        "labels",
        "token_slot_ids",
        "token_codebook_ids",
        "history_item_ids",
        "history_masks",
        "target_item_id",
        "history_semantic_ids",
        "target_semantic_ids",
        "seq_length",
    )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        stacked = {
            key: torch.stack([instance[key] for instance in instances], dim=0)
            for key in self.REQUIRED_STACK_KEYS
        }
        batch = {
            "input_ids": stacked["input_ids"],
            "attention_mask": stacked["attention_mask"],
            "labels": stacked["labels"],
            "token_slot_ids": stacked["token_slot_ids"],
            "token_codebook_ids": stacked["token_codebook_ids"],
            "history_item_ids": stacked["history_item_ids"],
            "history_masks": stacked["history_masks"],
            "target_item_ids": stacked["target_item_id"],
            "history_semantic_ids": stacked["history_semantic_ids"],
            "target_semantic_ids": stacked["target_semantic_ids"],
            "seq_lengths": stacked["seq_length"],
            # Keep aliases for ad-hoc scripts during the migration window.
            "target_item_id": stacked["target_item_id"],
            "seq_length": stacked["seq_length"],
        }
        for key in self.OPTIONAL_STACK_KEYS:
            if key in instances[0]:
                batch[key] = torch.stack([instance[key] for instance in instances], dim=0)
        return batch
