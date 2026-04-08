#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.fusion import fuse_embedding_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse aligned item embedding branches into a multimodal item embedding matrix."
    )
    parser.add_argument("--text-path", type=Path, default=None)
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--cf-path", type=Path, default=None)
    parser.add_argument("--vlm-path", type=Path, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="weighted_sum",
        choices=["weighted_sum", "concat", "mean"],
    )
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--vlm-weight", type=float, default=1.0)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--normalize-each", action="store_true")
    parser.add_argument("--normalize-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    named_paths = {
        "text": args.text_path,
        "image": args.image_path,
        "cf": args.cf_path,
        "vlm": args.vlm_path,
    }
    named_embeddings = {}
    for name, path in named_paths.items():
        if path is None:
            continue
        named_embeddings[name] = np.asarray(np.load(path), dtype=np.float32)

    if not named_embeddings:
        raise ValueError("At least one embedding branch must be provided.")

    weights = {
        "text": args.text_weight,
        "image": args.image_weight,
        "cf": args.cf_weight,
        "vlm": args.vlm_weight,
    }
    fused = fuse_embedding_dict(
        named_embeddings,
        strategy=args.strategy,
        weights=weights,
        normalize_each=args.normalize_each,
        normalize_output=args.normalize_output,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, fused.astype(np.float32, copy=False))

    manifest_path = args.manifest_path or args.output_path.with_suffix(".manifest.json")
    payload = {
        "strategy": args.strategy,
        "sources": {
            name: str(path)
            for name, path in named_paths.items()
            if path is not None
        },
        "weights": weights,
        "normalize_each": bool(args.normalize_each),
        "normalize_output": bool(args.normalize_output),
        "output_path": str(args.output_path),
        "shape": list(fused.shape),
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print("========== Multimodal Fusion Done ==========")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
