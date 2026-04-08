"""Raw dataset discovery helpers for the GenRec-style pipeline."""

from .amazon_music import AmazonMusicRawDataPaths, build_amazon_music_raw_manifest, resolve_amazon_music_raw_paths

__all__ = [
    "AmazonMusicRawDataPaths",
    "build_amazon_music_raw_manifest",
    "resolve_amazon_music_raw_paths",
]
