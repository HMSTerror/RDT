"""Fusion strategies over aligned item embedding matrices."""

from .strategies import FUSION_REGISTRY, concat_fusion, fuse_embedding_dict, weighted_sum_fusion

__all__ = [
    "FUSION_REGISTRY",
    "concat_fusion",
    "fuse_embedding_dict",
    "weighted_sum_fusion",
]
