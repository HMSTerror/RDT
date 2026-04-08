from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AmazonMusicRawDataPaths:
    dataset_root: Path
    reviews_path: Path
    meta_path: Path
    image_root: Path
    item_embeddings_path: Path | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        return {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in payload.items()
        }


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not resolve a valid Amazon Music path from candidates: "
        + ", ".join(str(path) for path in candidates)
    )


def resolve_amazon_music_raw_paths(
    dataset_root: str | Path = "data/Amazon_Music_And_Instruments",
    image_root: str | Path = "data/images",
    item_embeddings_path: str | Path | None = "data/Amazon_Music_And_Instruments/item_embeddings.npy",
) -> AmazonMusicRawDataPaths:
    dataset_root = Path(dataset_root)
    image_root = Path(image_root)

    reviews_path = _first_existing(
        [
            dataset_root / "Musical_Instruments_5.json",
            dataset_root / "Musical_Instruments_5.json.gz",
            dataset_root / "Digital_Music_5.json",
            dataset_root / "Digital_Music_5.json.gz",
        ]
    )
    meta_path = _first_existing(
        [
            dataset_root / "meta_Musical_Instruments.json",
            dataset_root / "meta_Musical_Instruments.json.gz",
            dataset_root / "meta_Digital_Music.json",
            dataset_root / "meta_Digital_Music.json.gz",
        ]
    )

    resolved_item_embeddings_path = None
    if item_embeddings_path is not None:
        candidate = Path(item_embeddings_path)
        if candidate.exists():
            resolved_item_embeddings_path = candidate

    return AmazonMusicRawDataPaths(
        dataset_root=dataset_root,
        reviews_path=reviews_path,
        meta_path=meta_path,
        image_root=image_root,
        item_embeddings_path=resolved_item_embeddings_path,
    )


def build_amazon_music_raw_manifest(
    output_path: str | Path,
    *,
    dataset_root: str | Path = "data/Amazon_Music_And_Instruments",
    image_root: str | Path = "data/images",
    item_embeddings_path: str | Path | None = "data/Amazon_Music_And_Instruments/item_embeddings.npy",
) -> dict:
    paths = resolve_amazon_music_raw_paths(
        dataset_root=dataset_root,
        image_root=image_root,
        item_embeddings_path=item_embeddings_path,
    )
    payload = {
        "dataset_name": "amazon_music",
        "raw_data": paths.to_dict(),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return payload
