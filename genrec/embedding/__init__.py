"""Embedding manifests and source metadata for the GenRec-style pipeline."""

from .manifest import EmbeddingArtifact, EmbeddingCollectionManifest, load_embedding_matrix, save_embedding_manifest

__all__ = [
    "EmbeddingArtifact",
    "EmbeddingCollectionManifest",
    "load_embedding_matrix",
    "save_embedding_manifest",
]
