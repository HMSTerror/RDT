from __future__ import annotations

import numpy as np

from .pq import ProductQuantizer


class OrthogonalProductQuantizer(ProductQuantizer):
    method_name = "opq"

    def __init__(
        self,
        *,
        num_subspaces: int = 4,
        codebook_size: int = 256,
        n_iters: int = 25,
        seed: int = 0,
    ) -> None:
        super().__init__(
            num_subspaces=num_subspaces,
            codebook_size=codebook_size,
            n_iters=n_iters,
            seed=seed,
        )
        self.mean_: np.ndarray | None = None
        self.rotation_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray) -> "OrthogonalProductQuantizer":
        embeddings = self._validate_embeddings(embeddings)
        self.mean_ = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - self.mean_

        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.rotation_ = vt.T.astype(np.float32, copy=False)
        rotated = centered @ self.rotation_
        super().fit(rotated)
        return self

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = self._validate_embeddings(embeddings)
        if self.mean_ is None or self.rotation_ is None:
            raise RuntimeError("OrthogonalProductQuantizer must be fitted before encode().")
        rotated = (embeddings - self.mean_) @ self.rotation_
        return super().encode(rotated)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.rotation_ is None:
            raise RuntimeError("OrthogonalProductQuantizer must be fitted before decode().")
        rotated_recon = super().decode(codes)
        return (rotated_recon @ self.rotation_.T + self.mean_).astype(np.float32, copy=False)
