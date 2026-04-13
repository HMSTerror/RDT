"""Generative backbone implementations for the GenRec-style pipeline."""

from .condition_projector import ConditionBranchProjector, ConditionProjectorOutput
from .conditioning_strategy import resolve_hybrid_conditioning_strategy
from .genrec_dit import GenRecDiTRunner, GenRecForwardOutput
from .genrec_hybrid_diffusion import GenRecHybridDiffusionRunner, HybridDiffusionOutput

__all__ = [
    "ConditionBranchProjector",
    "ConditionProjectorOutput",
    "resolve_hybrid_conditioning_strategy",
    "GenRecDiTRunner",
    "GenRecForwardOutput",
    "GenRecHybridDiffusionRunner",
    "HybridDiffusionOutput",
]
