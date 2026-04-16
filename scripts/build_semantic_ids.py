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
    parser.add_argument(
        "--fit-buffer-root",
        type=Path,
        default=None,
        help=(
            "Optional split buffer root used to fit the quantizer on train-only items "
            "while still encoding all items."
        ),
    )
    parser.add_argument(
        "--fit-buffer-split",
        type=str,
        default="train",
        help="Which split under --fit-buffer-root supplies the item set used for quantizer fitting.",
    )
    return parser.parse_args()


def load_item_map(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8-sig") as fp:
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


def resolve_split_root(buffer_root: Path, split_name: str) -> Path:
    candidate = buffer_root / split_name
    if (candidate / "stats.json").exists():
        return candidate
    if (buffer_root / "stats.json").exists():
        return buffer_root
    raise FileNotFoundError(f"Could not resolve split `{split_name}` under {buffer_root}.")


def collect_fit_item_indices(split_root: Path) -> np.ndarray:
    chunk_dirs = sorted(
        [
            path
            for path in split_root.glob("chunk_*")
            if path.is_dir() and (path / "samples.npz").exists()
        ],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not chunk_dirs:
        raise RuntimeError(f"No chunk directories found under {split_root}.")

    fit_item_ids: set[int] = set()
    for chunk_dir in chunk_dirs:
        with np.load(chunk_dir / "samples.npz", allow_pickle=False) as payload:
            target_item_ids = np.asarray(payload["target_item_ids"], dtype=np.int64)
            history_item_ids = np.asarray(payload["history_item_ids"], dtype=np.int64)
            history_mask = np.asarray(payload["history_mask"], dtype=np.bool_)

        fit_item_ids.update(int(item_idx) for item_idx in target_item_ids.tolist() if int(item_idx) >= 0)
        if history_item_ids.size > 0:
            valid_history_item_ids = history_item_ids[history_mask]
            fit_item_ids.update(int(item_idx) for item_idx in valid_history_item_ids.tolist() if int(item_idx) >= 0)

    if not fit_item_ids:
        raise RuntimeError(f"No valid item indices were found in split buffer {split_root}.")
    return np.asarray(sorted(fit_item_ids), dtype=np.int64)


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

    fit_item_indices = None
    fit_embeddings = embeddings
    fit_scope = "all_items"
    fit_buffer_root = None
    fit_buffer_split = None
    if args.fit_buffer_root is not None:
        fit_buffer_root = args.fit_buffer_root.resolve()
        fit_buffer_split = str(args.fit_buffer_split)
        split_root = resolve_split_root(fit_buffer_root, fit_buffer_split)
        fit_item_indices = collect_fit_item_indices(split_root)
        fit_embeddings = embeddings[fit_item_indices]
        fit_scope = f"buffer_split:{fit_buffer_split}"
        print(
            f"[fit] quantizer will fit on {fit_embeddings.shape[0]}/{embeddings.shape[0]} "
            f"items from {split_root}"
        )

    quantizer.fit(fit_embeddings)
    codes = quantizer.encode(embeddings)
    reconstructed = quantizer.decode(codes)
    reconstruction_mse_full = float(np.mean((embeddings - reconstructed) ** 2))
    reconstruction_mse_fit = float(np.mean((fit_embeddings - reconstructed[fit_item_indices]) ** 2)) if fit_item_indices is not None else reconstruction_mse_full

    result = {
        "codes": codes,
        "reconstructed": reconstructed,
        "metadata": {
            "method": args.method,
            "fit_scope": fit_scope,
            "fit_item_count": int(fit_embeddings.shape[0]),
            "reconstruction_mse_full": reconstruction_mse_full,
            "reconstruction_mse_fit": reconstruction_mse_fit,
            "fit_buffer_root": (str(fit_buffer_root) if fit_buffer_root is not None else None),
            "fit_buffer_split": fit_buffer_split,
        },
    }

    args.output_root.mkdir(parents=True, exist_ok=True)
    codes_path = args.output_root / "item_codes.npy"
    recon_path = args.output_root / "reconstructed_embeddings.npy"
    item_to_code_path = args.output_root / "item_to_code.json"
    code_to_items_path = args.output_root / "code_to_items.json"
    manifest_path = args.output_root / "quantizer_manifest.json"

    np.save(codes_path, result["codes"].astype(np.int32, copy=False))
    np.save(recon_path, result["reconstructed"].astype(np.float32, copy=False))

    idx_to_item = {idx: item_id for item_id, idx in item_map.items()}
    item_to_code = {}
    code_to_items = {}
    for idx in range(result["codes"].shape[0]):
        item_id = idx_to_item[idx]
        code = result["codes"][idx]
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
        "num_items": int(result["codes"].shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "code_shape": list(result["codes"].shape),
        **result["metadata"],
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
