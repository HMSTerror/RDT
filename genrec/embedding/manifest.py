from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


SUPPORTED_EMBEDDING_SOURCES = {"text", "image", "cf", "vlm", "fusion"}


@dataclass
class EmbeddingArtifact:
    name: str
    source_type: str
    path: str
    dim: int
    item_count: int | None = None
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.source_type not in SUPPORTED_EMBEDDING_SOURCES:
            raise ValueError(
                f"Unsupported embedding source_type `{self.source_type}`. "
                f"Supported values: {sorted(SUPPORTED_EMBEDDING_SOURCES)}."
            )


@dataclass
class EmbeddingCollectionManifest:
    dataset_name: str
    item_map_path: str | None
    artifacts: list[EmbeddingArtifact] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def load_embedding_matrix(path: str | Path) -> np.ndarray:
    matrix = np.load(Path(path), mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {tuple(matrix.shape)}.")
    return np.asarray(matrix, dtype=np.float32)


def save_embedding_manifest(manifest: EmbeddingCollectionManifest, output_path: str | Path) -> dict:
    for artifact in manifest.artifacts:
        artifact.validate()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return payload
