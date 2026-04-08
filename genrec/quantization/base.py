from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class QuantizationResult:
    codes: np.ndarray
    reconstructed: np.ndarray
    metadata: dict = field(default_factory=dict)


class BaseQuantizer:
    method_name = "base"

    def fit(self, embeddings: np.ndarray) -> "BaseQuantizer":
        raise NotImplementedError

    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def decode(self, codes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_encode(self, embeddings: np.ndarray) -> QuantizationResult:
        self.fit(embeddings)
        codes = self.encode(embeddings)
        reconstructed = self.decode(codes)
        mse = float(np.mean((np.asarray(embeddings, dtype=np.float32) - reconstructed) ** 2))
        return QuantizationResult(
            codes=codes,
            reconstructed=reconstructed,
            metadata={
                "method": self.method_name,
                "reconstruction_mse": mse,
            },
        )
