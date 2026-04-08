from __future__ import annotations

import numpy as np

from .base import BaseQuantizer
from .kmeans import fit_kmeans, squared_l2_distance


class RecursiveKMeansQuantizer(BaseQuantizer):
    method_name = "rkmeans"

    def __init__(
        self,
        *,
        levels: int = 4,
        branching_factor: int = 16,
        n_iters: int = 25,
        seed: int = 0,
    ) -> None:
        self.levels = int(levels)
        self.branching_factor = int(branching_factor)
        self.n_iters = int(n_iters)
        self.seed = int(seed)
        self.level_centroids: list[np.ndarray] = []
        self.leaf_lookup: dict[tuple[int, ...], np.ndarray] = {}
        self.fitted_codes_: np.ndarray | None = None
        self.embedding_dim_: int | None = None

    def fit(self, embeddings: np.ndarray) -> "RecursiveKMeansQuantizer":
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {tuple(embeddings.shape)}.")

        num_items, embedding_dim = embeddings.shape
        self.embedding_dim_ = int(embedding_dim)
        codes = np.zeros((num_items, self.levels), dtype=np.int32)
        self.level_centroids = []
        self.leaf_lookup = {}

        buckets = {(): np.arange(num_items, dtype=np.int32)}
        for level in range(self.levels):
            next_buckets: dict[tuple[int, ...], np.ndarray] = {}
            all_level_centroids = np.zeros((self.branching_factor, embedding_dim), dtype=np.float32)

            for bucket_offset, (prefix, item_indices) in enumerate(buckets.items()):
                subset = embeddings[item_indices]
                cluster_count = min(self.branching_factor, max(1, subset.shape[0]))
                centroids, labels, _ = fit_kmeans(
                    subset,
                    n_clusters=cluster_count,
                    n_iters=self.n_iters,
                    seed=self.seed + level + bucket_offset,
                )

                all_level_centroids[:cluster_count] = centroids

                for local_cluster in range(cluster_count):
                    mask = labels == local_cluster
                    if not np.any(mask):
                        continue
                    cluster_items = item_indices[mask]
                    codes[cluster_items, level] = local_cluster
                    child_prefix = prefix + (local_cluster,)
                    next_buckets[child_prefix] = cluster_items
                    self.leaf_lookup[child_prefix] = centroids[local_cluster]

            self.level_centroids.append(all_level_centroids)
            buckets = next_buckets if next_buckets else buckets

        self.fitted_codes_ = codes
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        if self.levels <= 0:
            raise ValueError("levels must be positive.")
        if not self.level_centroids:
            raise RuntimeError("RecursiveKMeansQuantizer must be fitted before encode().")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {tuple(embeddings.shape)}.")

        codes = np.zeros((embeddings.shape[0], self.levels), dtype=np.int32)
        for level in range(self.levels):
            distances = squared_l2_distance(embeddings, self.level_centroids[level])
            codes[:, level] = np.argmin(distances, axis=1).astype(np.int32)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=np.int32)
        if codes.ndim != 2 or codes.shape[1] != self.levels:
            raise ValueError(
                f"Expected codes with shape [N, {self.levels}], got {tuple(codes.shape)}."
            )
        if self.embedding_dim_ is None:
            raise RuntimeError("RecursiveKMeansQuantizer must be fitted before decode().")

        outputs = []
        for row in codes:
            prefix = tuple(int(value) for value in row.tolist())
            if prefix in self.leaf_lookup:
                outputs.append(self.leaf_lookup[prefix])
                continue

            fallback = None
            for length in range(self.levels - 1, -1, -1):
                shorter = prefix[:length]
                if shorter in self.leaf_lookup:
                    fallback = self.leaf_lookup[shorter]
                    break
            if fallback is None:
                fallback = np.zeros((self.embedding_dim_,), dtype=np.float32)
            outputs.append(fallback)
        return np.asarray(outputs, dtype=np.float32)
