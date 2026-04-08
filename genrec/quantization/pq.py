from __future__ import annotations

import numpy as np

from .base import BaseQuantizer
from .kmeans import assign_clusters, fit_kmeans


class ProductQuantizer(BaseQuantizer):
    method_name = "pq"

    def __init__(
        self,
        *,
        num_subspaces: int = 4,
        codebook_size: int = 256,
        n_iters: int = 25,
        seed: int = 0,
    ) -> None:
        self.num_subspaces = int(num_subspaces)
        self.codebook_size = int(codebook_size)
        self.n_iters = int(n_iters)
        self.seed = int(seed)
        self.subvector_dim = None
        self.codebooks: list[np.ndarray] = []

    def _validate_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape {tuple(embeddings.shape)}.")
        dim = embeddings.shape[1]
        if dim % self.num_subspaces != 0:
            raise ValueError(
                f"Embedding dim {dim} must be divisible by num_subspaces={self.num_subspaces}."
            )
        return embeddings

    def fit(self, embeddings: np.ndarray) -> "ProductQuantizer":
        embeddings = self._validate_embeddings(embeddings)
        self.subvector_dim = embeddings.shape[1] // self.num_subspaces
        self.codebooks = []

        for subspace_idx in range(self.num_subspaces):
            start = subspace_idx * self.subvector_dim
            end = start + self.subvector_dim
            subspace = embeddings[:, start:end]
            centroids, _, _ = fit_kmeans(
                subspace,
                n_clusters=self.codebook_size,
                n_iters=self.n_iters,
                seed=self.seed + subspace_idx,
            )
            self.codebooks.append(centroids)
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = self._validate_embeddings(embeddings)
        if not self.codebooks:
            raise RuntimeError("ProductQuantizer must be fitted before calling encode().")

        codes = np.empty((embeddings.shape[0], self.num_subspaces), dtype=np.int32)
        for subspace_idx, codebook in enumerate(self.codebooks):
            start = subspace_idx * self.subvector_dim
            end = start + self.subvector_dim
            subspace = embeddings[:, start:end]
            codes[:, subspace_idx] = assign_clusters(subspace, codebook)
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if not self.codebooks or self.subvector_dim is None:
            raise RuntimeError("ProductQuantizer must be fitted before calling decode().")

        codes = np.asarray(codes, dtype=np.int32)
        if codes.ndim != 2 or codes.shape[1] != self.num_subspaces:
            raise ValueError(
                f"Expected codes with shape [N, {self.num_subspaces}], got {tuple(codes.shape)}."
            )

        chunks = []
        for subspace_idx, codebook in enumerate(self.codebooks):
            chunks.append(codebook[codes[:, subspace_idx]])
        return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
