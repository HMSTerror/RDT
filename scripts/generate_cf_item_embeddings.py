#!/usr/bin/env python
# coding=utf-8

"""
Generate collaborative item embeddings aligned with preprocess_amazon.py item order.

Supported methods:
- item_item_sppmi:
    item-item co-occurrence within a sliding window, followed by shifted-PPMI
    and low-rank projection
- item2vec:
    skip-gram with negative sampling over item sequences
- mf_bpr:
    implicit matrix factorization trained with BPR loss on user-item pairs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from preprocess_amazon import (  # noqa: E402
    EMBED_DIM,
    K_CORE,
    apply_iterative_k_core,
    build_contiguous_mappings,
    build_user_sequences,
    load_review_interactions,
)


SUPPORTED_METHODS = ("item_item_sppmi", "item2vec", "mf_bpr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate collaborative item embeddings aligned with preprocess_amazon.py "
            "item ordering."
        )
    )
    parser.add_argument(
        "--reviews-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/Musical_Instruments_5.json"),
        help="Path to the raw Amazon review JSON / JSON.GZ file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/Amazon_Music_And_Instruments/item_embeddings_cf.npy"),
        help="Where to save the generated [num_items, 128] float32 CF embedding matrix.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="item2vec",
        choices=list(SUPPORTED_METHODS),
        help="Collaborative embedding method.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=EMBED_DIM,
        help="Final embedding dimension. RecSys-DiT expects 128.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Sliding-window size used by item-item and item2vec methods.",
    )
    parser.add_argument(
        "--negative-samples",
        type=int,
        default=10,
        help="Number of negatives for item2vec and mf_bpr.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs for item2vec / mf_bpr.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Mini-batch size for item2vec / mf_bpr.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for item2vec / mf_bpr.",
    )
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1e-6,
        help="L2 regularization weight for mf_bpr.",
    )
    parser.add_argument(
        "--sppmi-shift",
        type=float,
        default=5.0,
        help="Shift constant k used in SPPMI = max(PPMI - log(k), 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use, e.g. cuda, cuda:0, or cpu. Defaults to cuda if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file if it already exists.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_aligned_sequences(
    reviews_path: Path,
) -> tuple[List[Tuple[str, str, int]], Dict[str, int], Dict[str, int], Dict[int, List[str]], List[str]]:
    interactions = load_review_interactions(reviews_path)
    if not interactions:
        raise RuntimeError("No valid interactions were found in the review file.")
    print(f"[reviews] raw interactions: {len(interactions)}")

    filtered = apply_iterative_k_core(interactions, min_count=K_CORE)
    if not filtered:
        raise RuntimeError("5-core filtering removed all interactions.")

    user_map, item_map = build_contiguous_mappings(filtered)
    sequences = build_user_sequences(filtered, user_map)
    ordered_item_ids = [
        item_id for item_id, _ in sorted(item_map.items(), key=lambda pair: pair[1])
    ]
    print(
        f"[mapping] aligned users={len(user_map)} items={len(item_map)} "
        f"sequences={len(sequences)}"
    )
    return filtered, user_map, item_map, sequences, ordered_item_ids


def convert_sequences_to_indices(
    sequences: Dict[int, List[str]],
    item_map: Dict[str, int],
) -> List[List[int]]:
    indexed_sequences: List[List[int]] = []
    for sequence in sequences.values():
        indexed_sequences.append([item_map[item_id] for item_id in sequence if item_id in item_map])
    return indexed_sequences


def reduce_to_target_dim(embeddings: torch.Tensor, output_dim: int, seed: int) -> torch.Tensor:
    num_items, raw_dim = embeddings.shape
    if raw_dim == output_dim:
        return F.normalize(embeddings, dim=-1)

    if raw_dim < output_dim:
        padded = torch.zeros(num_items, output_dim, dtype=embeddings.dtype)
        padded[:, :raw_dim] = embeddings
        return F.normalize(padded, dim=-1)

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    max_rank = min(num_items, raw_dim)
    effective_dim = min(output_dim, max_rank)

    torch.manual_seed(seed)
    q = min(max(effective_dim + 8, effective_dim), max_rank)
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    reduced = centered @ v[:, :effective_dim]

    if effective_dim < output_dim:
        padded = torch.zeros(num_items, output_dim, dtype=reduced.dtype)
        padded[:, :effective_dim] = reduced
        reduced = padded

    return F.normalize(reduced, dim=-1)


def build_item_item_sppmi_embeddings(
    indexed_sequences: Sequence[Sequence[int]],
    *,
    num_items: int,
    window_size: int,
    output_dim: int,
    sppmi_shift: float,
    seed: int,
) -> torch.Tensor:
    print("[item_item_sppmi] building co-occurrence matrix...")
    cooc = np.zeros((num_items, num_items), dtype=np.float32)

    for sequence in tqdm(indexed_sequences, desc="Counting item-item pairs", unit="seq"):
        seq_len = len(sequence)
        for center_pos, center_item in enumerate(sequence):
            left = max(0, center_pos - window_size)
            right = min(seq_len, center_pos + window_size + 1)
            for ctx_pos in range(left, right):
                if ctx_pos == center_pos:
                    continue
                context_item = sequence[ctx_pos]
                distance = abs(ctx_pos - center_pos)
                weight = 1.0 / max(1, distance)
                cooc[center_item, context_item] += weight

    total = float(cooc.sum())
    if total <= 0:
        raise RuntimeError("Co-occurrence counting produced an empty matrix.")

    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    denom = np.clip(row_sum * col_sum, 1e-12, None)
    numer = np.clip(cooc * total, 1e-12, None)
    sppmi = np.log(numer / denom) - math.log(max(sppmi_shift, 1e-6))
    sppmi = np.maximum(sppmi, 0.0).astype(np.float32, copy=False)

    sppmi_tensor = torch.from_numpy(sppmi)
    reduced = reduce_to_target_dim(sppmi_tensor, output_dim=output_dim, seed=seed)
    return reduced.cpu()


def build_skipgram_pairs(
    indexed_sequences: Sequence[Sequence[int]],
    *,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers: List[int] = []
    contexts: List[int] = []
    for sequence in indexed_sequences:
        seq_len = len(sequence)
        for center_pos, center_item in enumerate(sequence):
            left = max(0, center_pos - window_size)
            right = min(seq_len, center_pos + window_size + 1)
            for ctx_pos in range(left, right):
                if ctx_pos == center_pos:
                    continue
                centers.append(center_item)
                contexts.append(sequence[ctx_pos])
    if not centers:
        raise RuntimeError("No skip-gram pairs were generated.")
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


class Item2VecModel(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_embeddings = nn.Embedding(num_items, embedding_dim)
        self.output_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.input_embeddings.weight, std=0.02)
        nn.init.zeros_(self.output_embeddings.weight)

    def forward(self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        center_vec = self.input_embeddings(centers)
        context_vec = self.output_embeddings(contexts)
        positive_scores = (center_vec * context_vec).sum(dim=-1)
        positive_loss = -F.logsigmoid(positive_scores)

        negative_vec = self.output_embeddings(negatives)
        negative_scores = torch.bmm(negative_vec, center_vec.unsqueeze(-1)).squeeze(-1)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=-1)
        return (positive_loss + negative_loss).mean()


def build_item2vec_embeddings(
    indexed_sequences: Sequence[Sequence[int]],
    *,
    num_items: int,
    embedding_dim: int,
    window_size: int,
    negative_samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    centers_np, contexts_np = build_skipgram_pairs(indexed_sequences, window_size=window_size)
    unigram_counts = np.bincount(contexts_np, minlength=num_items).astype(np.float64)
    noise_distribution = np.power(np.clip(unigram_counts, 1.0, None), 0.75)
    noise_distribution /= np.clip(noise_distribution.sum(), 1e-12, None)

    model = Item2VecModel(num_items=num_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    num_pairs = centers_np.shape[0]
    print(f"[item2vec] training pairs={num_pairs}")
    for epoch in range(epochs):
        permutation = rng.permutation(num_pairs)
        centers_np = centers_np[permutation]
        contexts_np = contexts_np[permutation]
        epoch_loss = 0.0
        num_batches = 0

        iterator = range(0, num_pairs, batch_size)
        for start in tqdm(iterator, desc=f"item2vec epoch {epoch + 1}/{epochs}", unit="batch"):
            end = min(start + batch_size, num_pairs)
            centers = torch.from_numpy(centers_np[start:end]).to(device=device, dtype=torch.long)
            contexts = torch.from_numpy(contexts_np[start:end]).to(device=device, dtype=torch.long)
            negatives_np = rng.choice(
                num_items,
                size=(end - start, negative_samples),
                replace=True,
                p=noise_distribution,
            )
            negatives = torch.from_numpy(negatives_np).to(device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            loss = model(centers, contexts, negatives)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        print(f"[item2vec] epoch={epoch + 1} avg_loss={epoch_loss / max(1, num_batches):.6f}")

    embeddings = model.input_embeddings.weight.detach().float().cpu()
    return F.normalize(embeddings, dim=-1)


class BPRMFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.normal_(self.user_embeddings.weight, std=0.02)
        nn.init.normal_(self.item_embeddings.weight, std=0.02)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user_vec = self.user_embeddings(users)
        pos_vec = self.item_embeddings(pos_items)
        neg_vec = self.item_embeddings(neg_items)
        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)

        pos_scores = (user_vec * pos_vec).sum(dim=-1) + pos_bias
        neg_scores = (user_vec * neg_vec).sum(dim=-1) + neg_bias
        return pos_scores, neg_scores


def sample_bpr_negatives(
    users: np.ndarray,
    *,
    num_items: int,
    user_positive_sets: Dict[int, set[int]],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = np.empty_like(users, dtype=np.int64)
    for idx, user in enumerate(users.tolist()):
        positive_set = user_positive_sets[int(user)]
        candidate = int(rng.integers(0, num_items))
        while candidate in positive_set:
            candidate = int(rng.integers(0, num_items))
        negatives[idx] = candidate
    return negatives


def build_mf_bpr_embeddings(
    interactions: Sequence[Tuple[str, str, int]],
    *,
    user_map: Dict[str, int],
    item_map: Dict[str, int],
    embedding_dim: int,
    negative_samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    reg_weight: float,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    user_item_pairs = np.asarray(
        [(user_map[user_id], item_map[item_id]) for user_id, item_id, _ in interactions],
        dtype=np.int64,
    )
    users_np = user_item_pairs[:, 0]
    items_np = user_item_pairs[:, 1]

    user_positive_sets: Dict[int, set[int]] = defaultdict(set)
    for user_idx, item_idx in user_item_pairs.tolist():
        user_positive_sets[int(user_idx)].add(int(item_idx))

    model = BPRMFModel(
        num_users=len(user_map),
        num_items=len(item_map),
        embedding_dim=embedding_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)

    num_pairs = user_item_pairs.shape[0]
    print(f"[mf_bpr] training pairs={num_pairs}")
    for epoch in range(epochs):
        permutation = rng.permutation(num_pairs)
        users_shuffled = users_np[permutation]
        items_shuffled = items_np[permutation]
        epoch_loss = 0.0
        num_batches = 0

        iterator = range(0, num_pairs, batch_size)
        for start in tqdm(iterator, desc=f"mf_bpr epoch {epoch + 1}/{epochs}", unit="batch"):
            end = min(start + batch_size, num_pairs)
            batch_users_np = users_shuffled[start:end]
            batch_pos_np = items_shuffled[start:end]

            batch_users = torch.from_numpy(batch_users_np).to(device=device, dtype=torch.long)
            batch_pos = torch.from_numpy(batch_pos_np).to(device=device, dtype=torch.long)

            total_loss = 0.0
            for _ in range(max(1, negative_samples)):
                batch_neg_np = sample_bpr_negatives(
                    batch_users_np,
                    num_items=len(item_map),
                    user_positive_sets=user_positive_sets,
                    rng=rng,
                )
                batch_neg = torch.from_numpy(batch_neg_np).to(device=device, dtype=torch.long)

                pos_scores, neg_scores = model(batch_users, batch_pos, batch_neg)
                bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                reg_loss = reg_weight * (
                    model.user_embeddings(batch_users).pow(2).mean()
                    + model.item_embeddings(batch_pos).pow(2).mean()
                    + model.item_embeddings(batch_neg).pow(2).mean()
                )
                total_loss = total_loss + bpr_loss + reg_loss

            total_loss = total_loss / max(1, negative_samples)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            num_batches += 1

        print(f"[mf_bpr] epoch={epoch + 1} avg_loss={epoch_loss / max(1, num_batches):.6f}")

    embeddings = model.item_embeddings.weight.detach().float().cpu()
    return F.normalize(embeddings, dim=-1)


def save_sidecar_metadata(
    output_path: Path,
    *,
    args: argparse.Namespace,
    num_users: int,
    num_items: int,
    num_sequences: int,
    num_interactions: int,
) -> None:
    meta_payload = {
        "reviews_path": str(args.reviews_path),
        "output_path": str(output_path),
        "method": args.method,
        "output_dim": int(args.output_dim),
        "window_size": int(args.window_size),
        "negative_samples": int(args.negative_samples),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "reg_weight": float(args.reg_weight),
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_sequences": int(num_sequences),
        "num_interactions": int(num_interactions),
        "k_core": int(K_CORE),
        "embedding_type": f"collaborative_{args.method}",
    }
    sidecar_path = output_path.with_suffix(".meta.json")
    with open(sidecar_path, "w", encoding="utf-8") as fp:
        json.dump(meta_payload, fp, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    if not args.reviews_path.exists():
        raise FileNotFoundError(f"Review file not found: {args.reviews_path}")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_path}. "
            "Pass --overwrite to replace it."
        )

    print("========== Generate CF Item Embeddings ==========")
    print(f"reviews_path          : {args.reviews_path}")
    print(f"output_path           : {args.output_path}")
    print(f"method                : {args.method}")
    print(f"device                : {device}")
    print(f"window_size           : {args.window_size}")
    print(f"negative_samples      : {args.negative_samples}")
    print(f"epochs                : {args.epochs}")
    print(f"batch_size            : {args.batch_size}")
    print(f"learning_rate         : {args.learning_rate}")

    filtered, user_map, item_map, sequences, _ = build_aligned_sequences(args.reviews_path)
    indexed_sequences = convert_sequences_to_indices(sequences, item_map)

    if args.method == "item_item_sppmi":
        embeddings = build_item_item_sppmi_embeddings(
            indexed_sequences,
            num_items=len(item_map),
            window_size=args.window_size,
            output_dim=args.output_dim,
            sppmi_shift=args.sppmi_shift,
            seed=args.seed,
        )
    elif args.method == "item2vec":
        embeddings = build_item2vec_embeddings(
            indexed_sequences,
            num_items=len(item_map),
            embedding_dim=args.output_dim,
            window_size=args.window_size,
            negative_samples=args.negative_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            seed=args.seed,
        )
    elif args.method == "mf_bpr":
        embeddings = build_mf_bpr_embeddings(
            filtered,
            user_map=user_map,
            item_map=item_map,
            embedding_dim=args.output_dim,
            negative_samples=args.negative_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            reg_weight=args.reg_weight,
            device=device,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, embeddings.cpu().numpy().astype(np.float32, copy=False))
    save_sidecar_metadata(
        args.output_path,
        args=args,
        num_users=len(user_map),
        num_items=len(item_map),
        num_sequences=len(sequences),
        num_interactions=len(filtered),
    )

    print(f"[done] wrote {tuple(embeddings.shape)} float32 embeddings to {args.output_path}")
    print(f"[done] wrote sidecar metadata to {args.output_path.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main()
