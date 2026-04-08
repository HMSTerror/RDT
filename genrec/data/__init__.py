"""Tokenized data interfaces for the GenRec-style pipeline."""

from .tokenized_dataset import GenRecTokenizedCollator, GenRecTokenizedDataset

__all__ = [
    "GenRecTokenizedCollator",
    "GenRecTokenizedDataset",
]
