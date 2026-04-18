#!/usr/bin/env python
# coding=utf-8

"""
Align item embedding tables into a target embedding space.

Typical use in this repo:
- source = text item embeddings or image item embeddings
- target = CF item embeddings

The script trains a small MLP adapter on item-level pairs:
    aligned(source[item]) ~= target[item]

It writes:
- transformed full item table to --output-path
- training metadata to --output-path.with_suffix(".meta.json")
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Align one item embedding table into another embedding space."
    )
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--target-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--source-name", type=str, default="source")
    parser.add_argument("--target-name", type=str, default="target")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-hidden-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--cosine-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
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


def load_table(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Embedding table not found: {path}")
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embedding table at {path}, got shape {tuple(array.shape)}")
    return array


class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(max(0, num_hidden_layers))
        )
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.residual_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.in_proj(x))
        for layer in self.hidden_layers:
            hidden = hidden + layer(hidden)
        return self.out_proj(hidden) + self.residual_proj(x)


def build_split_indices(num_items: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_items < 2:
        raise ValueError("Need at least 2 items to create train/val split.")
    generator = np.random.default_rng(seed)
    indices = np.arange(num_items, dtype=np.int64)
    generator.shuffle(indices)
    val_size = int(round(num_items * val_ratio))
    val_size = min(max(val_size, 1), num_items - 1)
    val_idx = np.sort(indices[:val_size])
    train_idx = np.sort(indices[val_size:])
    return train_idx, val_idx


def iterate_minibatches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[np.ndarray]:
    working = np.array(indices, copy=True)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(working)
    return [working[start : start + batch_size] for start in range(0, len(working), batch_size)]


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    mse = F.mse_loss(pred, target)
    cosine = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()

    total = mse_weight * mse + cosine_weight * cosine
    metrics = {
        "mse": float(mse.detach().item()),
        "cosine": float(cosine.detach().item()),
    }

    if contrastive_weight > 0.0:
        logits = (pred @ target.t()) / contrastive_temperature
        labels = torch.arange(pred.shape[0], device=pred.device)
        contrastive = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )
        total = total + contrastive_weight * contrastive
        metrics["contrastive"] = float(contrastive.detach().item())
    else:
        metrics["contrastive"] = 0.0

    metrics["total"] = float(total.detach().item())
    return total, metrics


def evaluate_model(
    model: nn.Module,
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "mse": 0.0, "cosine": 0.0, "contrastive": 0.0}
    count = 0

    with torch.no_grad():
        for batch_idx in iterate_minibatches(indices, batch_size, shuffle=False, seed=0):
            source = source_tensor[batch_idx]
            target = target_tensor[batch_idx]
            _, metrics = compute_loss(
                model(source),
                target,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
            )
            batch_n = int(len(batch_idx))
            count += batch_n
            for key in totals:
                totals[key] += metrics[key] * batch_n

    return {key: (value / max(count, 1)) for key, value in totals.items()}


def train_adapter(
    source_table: np.ndarray,
    target_table: np.ndarray,
    *,
    hidden_dim: int,
    num_hidden_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_ratio: float,
    patience: int,
    mse_weight: float,
    cosine_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    device: torch.device,
    seed: int,
) -> tuple[nn.Module, dict[str, object]]:
    source_tensor = torch.from_numpy(source_table).to(device=device, dtype=torch.float32)
    target_tensor = torch.from_numpy(target_table).to(device=device, dtype=torch.float32)

    train_idx, val_idx = build_split_indices(source_table.shape[0], val_ratio, seed)
    model = ResidualMLP(
        input_dim=source_table.shape[1],
        output_dim=target_table.shape[1],
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    epochs_without_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "mse": 0.0, "cosine": 0.0, "contrastive": 0.0}
        seen = 0

        for step, batch_idx in enumerate(
            iterate_minibatches(train_idx, batch_size, shuffle=True, seed=seed + epoch),
            start=1,
        ):
            source = source_tensor[batch_idx]
            target = target_tensor[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_loss(
                model(source),
                target,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
            )
            loss.backward()
            optimizer.step()

            batch_n = int(len(batch_idx))
            seen += batch_n
            for key in running:
                running[key] += metrics[key] * batch_n

        train_metrics = {key: (value / max(seen, 1)) for key, value in running.items()}
        val_metrics = evaluate_model(
            model,
            source_tensor,
            target_tensor,
            val_idx,
            batch_size=batch_size,
            mse_weight=mse_weight,
            cosine_weight=cosine_weight,
            contrastive_weight=contrastive_weight,
            contrastive_temperature=contrastive_temperature,
        )
        epoch_record = {
            "epoch": float(epoch),
            "train_total": train_metrics["total"],
            "train_mse": train_metrics["mse"],
            "train_cosine": train_metrics["cosine"],
            "train_contrastive": train_metrics["contrastive"],
            "val_total": val_metrics["total"],
            "val_mse": val_metrics["mse"],
            "val_cosine": val_metrics["cosine"],
            "val_contrastive": val_metrics["contrastive"],
        }
        history.append(epoch_record)

        print(
            f"[epoch {epoch:03d}] "
            f"train_total={train_metrics['total']:.6f} "
            f"val_total={val_metrics['total']:.6f} "
            f"val_mse={val_metrics['mse']:.6f} "
            f"val_cos={val_metrics['cosine']:.6f}"
        )

        if val_metrics["total"] + 1e-8 < best_val:
            best_val = val_metrics["total"]
            best_state = deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[early-stop] patience={patience} reached at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    meta = {
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "best_val_total": float(best_val),
        "history": history,
    }
    return model, meta


@torch.no_grad()
def transform_all(model: nn.Module, source_table: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    source_tensor = torch.from_numpy(source_table).to(device=device, dtype=torch.float32)
    outputs = []
    for start in range(0, source_tensor.shape[0], batch_size):
        batch = source_tensor[start : start + batch_size]
        outputs.append(F.normalize(model(batch), dim=-1).cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output_path}. Pass --overwrite to replace it."
        )

    source_table = load_table(args.source_path)
    target_table = load_table(args.target_path)
    if source_table.shape[0] != target_table.shape[0]:
        raise ValueError(
            "Source and target tables must share the same item ordering and number of items: "
            f"{source_table.shape[0]} vs {target_table.shape[0]}"
        )

    print("========== Align Item Embeddings ==========")
    print(f"source_path           : {args.source_path}")
    print(f"target_path           : {args.target_path}")
    print(f"output_path           : {args.output_path}")
    print(f"source_name           : {args.source_name}")
    print(f"target_name           : {args.target_name}")
    print(f"num_items             : {source_table.shape[0]}")
    print(f"source_dim            : {source_table.shape[1]}")
    print(f"target_dim            : {target_table.shape[1]}")
    print(f"device                : {device}")

    model, training_meta = train_adapter(
        source_table,
        target_table,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        patience=args.patience,
        mse_weight=args.mse_weight,
        cosine_weight=args.cosine_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        device=device,
        seed=args.seed,
    )
    aligned_table = transform_all(model, source_table, args.batch_size, device)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path, aligned_table)

    payload = {
        "source_path": str(args.source_path),
        "target_path": str(args.target_path),
        "output_path": str(args.output_path),
        "source_name": args.source_name,
        "target_name": args.target_name,
        "num_items": int(source_table.shape[0]),
        "source_dim": int(source_table.shape[1]),
        "target_dim": int(target_table.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "num_hidden_layers": int(args.num_hidden_layers),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "val_ratio": float(args.val_ratio),
        "patience": int(args.patience),
        "mse_weight": float(args.mse_weight),
        "cosine_weight": float(args.cosine_weight),
        "contrastive_weight": float(args.contrastive_weight),
        "contrastive_temperature": float(args.contrastive_temperature),
        "seed": int(args.seed),
        "training": training_meta,
    }
    meta_path = args.output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)

    print(f"[done] wrote aligned embeddings to {args.output_path}")
    print(f"[done] wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
