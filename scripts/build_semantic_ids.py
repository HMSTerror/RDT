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

from genrec.quantization import build_quantizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic IDs from dense item embeddings using a selected quantizer."
    )
    parser.add_argument("--embeddings-path", type=Path, required=True)
    parser.add_argument("--item-map-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="pq",
        choices=["pq", "opq", "rkmeans", "rqvae"],
    )
    parser.add_argument("--num-subspaces", type=int, default=4)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--branching-factor", type=int, default=16)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_item_map(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {str(key): int(value) for key, value in payload.items()}


def code_to_string(code: np.ndarray) -> str:
    return "-".join(str(int(x)) for x in code.tolist())


def build_method_kwargs(args: argparse.Namespace) -> dict:
    if args.method in {"pq", "opq"}:
        return {
            "num_subspaces": args.num_subspaces,
            "codebook_size": args.codebook_size,
            "n_iters": args.max_iter,
            "seed": args.seed,
        }
    if args.method == "rkmeans":
        return {
            "levels": args.levels,
            "branching_factor": args.branching_factor,
            "n_iters": args.max_iter,
            "seed": args.seed,
        }
    if args.method == "rqvae":
        return {
            "num_codebooks": args.num_subspaces,
            "codebook_size": args.codebook_size,
        }
    raise ValueError(f"Unsupported method: {args.method}")


def main() -> None:
    args = parse_args()

    embeddings = np.asarray(np.load(args.embeddings_path), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected `embeddings_path` to contain a 2D matrix, got {tuple(embeddings.shape)}."
        )
    non_finite = int(np.count_nonzero(~np.isfinite(embeddings)))
    if non_finite > 0:
        print(
            f"[warn] Found {non_finite} non-finite values in embeddings. "
            "Replacing NaN/Inf with 0 before quantization."
        )
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    item_map = load_item_map(args.item_map_path)
    if embeddings.shape[0] != len(item_map):
        raise ValueError(
            "Mismatch between embedding rows and item_map size: "
            f"{embeddings.shape[0]} vs {len(item_map)}."
        )

    quantizer = build_quantizer(args.method, **build_method_kwargs(args))
    result = quantizer.fit_encode(embeddings)

    args.output_root.mkdir(parents=True, exist_ok=True)
    codes_path = args.output_root / "item_codes.npy"
    recon_path = args.output_root / "reconstructed_embeddings.npy"
    item_to_code_path = args.output_root / "item_to_code.json"
    code_to_items_path = args.output_root / "code_to_items.json"
    manifest_path = args.output_root / "quantizer_manifest.json"

    np.save(codes_path, result.codes.astype(np.int32, copy=False))
    np.save(recon_path, result.reconstructed.astype(np.float32, copy=False))

    idx_to_item = {idx: item_id for item_id, idx in item_map.items()}
    item_to_code = {}
    code_to_items = {}
    for idx in range(result.codes.shape[0]):
        item_id = idx_to_item[idx]
        code = result.codes[idx]
        code_list = [int(value) for value in code.tolist()]
        item_to_code[item_id] = code_list
        key = code_to_string(code)
        code_to_items.setdefault(key, []).append(item_id)

    with open(item_to_code_path, "w", encoding="utf-8") as fp:
        json.dump(item_to_code, fp, indent=2, ensure_ascii=False)
    with open(code_to_items_path, "w", encoding="utf-8") as fp:
        json.dump(code_to_items, fp, indent=2, ensure_ascii=False)

    manifest = {
        "method": args.method,
        "embeddings_path": str(args.embeddings_path),
        "item_map_path": str(args.item_map_path),
        "output_root": str(args.output_root),
        "num_items": int(result.codes.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "code_shape": list(result.codes.shape),
        **result.metadata,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print("========== Semantic IDs Built ==========")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"saved_codes          = {codes_path}")
    print(f"saved_item_to_code   = {item_to_code_path}")
    print(f"saved_code_to_items  = {code_to_items_path}")
    print(f"saved_reconstruction = {recon_path}")


if __name__ == "__main__":
    main()
