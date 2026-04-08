from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _stringify(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    return value


@dataclass
class RawDataStageManifest:
    dataset_name: str
    dataset_root: str
    reviews_path: str
    meta_path: str
    image_root: str
    item_embeddings_path: str | None = None


@dataclass
class PreprocessingStageManifest:
    output_root: str
    split_mode: str
    history_len: int
    chunk_size: int
    stats_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingStageManifest:
    source_name: str
    source_type: str
    matrix_path: str
    dim: int
    item_count: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class FusionStageManifest:
    enabled: bool
    strategy: str
    source_names: list[str] = field(default_factory=list)
    output_path: str | None = None
    output_dim: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class QuantizationStageManifest:
    method: str
    input_path: str
    output_root: str
    code_shape: list[int] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GenRecPipelineManifest:
    raw_data: RawDataStageManifest
    preprocessing: PreprocessingStageManifest
    embeddings: list[EmbeddingStageManifest] = field(default_factory=list)
    fusion: FusionStageManifest | None = None
    quantization: QuantizationStageManifest | None = None

    def to_dict(self) -> dict:
        return _stringify(asdict(self))


def save_pipeline_manifest(manifest: GenRecPipelineManifest, output_path: str | Path) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return payload
