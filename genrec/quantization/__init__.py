"""Quantizers for building semantic IDs from dense item embeddings."""

from .opq import OrthogonalProductQuantizer
from .pq import ProductQuantizer
from .rkmeans import RecursiveKMeansQuantizer
from .rqvae import RQVAEQuantizer


def build_quantizer(method: str, **kwargs):
    method = method.lower()
    if method == "pq":
        return ProductQuantizer(**kwargs)
    if method == "opq":
        return OrthogonalProductQuantizer(**kwargs)
    if method == "rkmeans":
        return RecursiveKMeansQuantizer(**kwargs)
    if method == "rqvae":
        return RQVAEQuantizer(**kwargs)
    raise ValueError(
        f"Unsupported quantization method `{method}`. "
        "Available methods: ['opq', 'pq', 'rkmeans', 'rqvae']."
    )


__all__ = [
    "OrthogonalProductQuantizer",
    "ProductQuantizer",
    "RecursiveKMeansQuantizer",
    "RQVAEQuantizer",
    "build_quantizer",
]
