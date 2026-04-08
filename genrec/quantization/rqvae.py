from __future__ import annotations

import numpy as np

from .base import BaseQuantizer


class RQVAEQuantizer(BaseQuantizer):
    method_name = "rqvae"

    def __init__(self, *, num_codebooks: int = 4, codebook_size: int = 256) -> None:
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)

    def fit(self, embeddings: np.ndarray) -> "RQVAEQuantizer":
        raise NotImplementedError(
            "RQVAEQuantizer is only a placeholder interface in this stage. "
            "A full RQ-VAE implementation requires a dedicated training loop, "
            "codebook losses, and checkpoint management."
        )

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RQVAEQuantizer has not been implemented yet.")

    def decode(self, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RQVAEQuantizer has not been implemented yet.")
