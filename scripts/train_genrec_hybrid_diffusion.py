#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers.optimization import get_scheduler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.data import GenRecTokenizedCollator, GenRecTokenizedDataset  # noqa: E402
from genrec.models import GenRecHybridDiffusionRunner, resolve_hybrid_conditioning_strategy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Hybrid GenRec Diffusion model on tokenized train/val/test buffers."
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("configs/genrec_hybrid_diffusion_amazon.yaml"),
        help="YAML config for the hybrid diffusion experiment.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints and run metadata.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_every_epochs", type=int, default=None)
    parser.add_argument("--save_every_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory to resume from (e.g., output_dir/checkpoint-43000).",
    )
    parser.add_argument(
        "--load_model_only_from_checkpoint",
        type=Path,
        default=None,
        help=(
            "Warm-start from a checkpoint by loading model weights only. "
            "Optimizer/scheduler/global_step are reset so training restarts from epoch 0."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode passed to Accelerator.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def infer_embedding_dim(path_like: str | None) -> int | None:
    if not path_like:
        return None
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Condition embedding path not found: {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix at {path}, got shape {tuple(array.shape)}.")
    return int(array.shape[1])


def load_item_latent_table(path_like: str | Path) -> torch.Tensor:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Target latent table not found: {path}")
    array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [num_items, latent_dim] latent table, got shape {tuple(array.shape)}.")
    return torch.from_numpy(array.copy())


def make_dataloader(
    dataset: GenRecTokenizedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=GenRecTokenizedCollator(),
    )


def maybe_build_fixed_eval_subset(
    dataset: GenRecTokenizedDataset | None,
    *,
    subset_size: int | None,
    subset_offset: int,
) -> GenRecTokenizedDataset | Subset | None:
    if dataset is None or subset_size is None:
        return dataset
    if subset_size <= 0:
        return dataset

    start = max(0, int(subset_offset))
    end = min(len(dataset), start + int(subset_size))
    if start >= len(dataset) or start >= end:
        raise ValueError(
            f"Invalid validation subset range: start={start}, end={end}, dataset_size={len(dataset)}."
        )
    return Subset(dataset, list(range(start, end)))


def maybe_print(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_local_main_process:
        print(message)


def resolve_training_value(cli_value, config_section: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def resolve_auxiliary_weight_for_epoch(
    *,
    default_weight: float,
    curriculum_cfg: dict,
    epoch_value: float,
    modality_key: str,
) -> float:
    if not curriculum_cfg or not bool(curriculum_cfg.get("enabled", False)):
        return float(default_weight)

    high_epochs = int(curriculum_cfg.get("high_epochs", 0))
    shared_high_weight = curriculum_cfg.get("high_weight")
    shared_low_weight = curriculum_cfg.get("low_weight")
    modality_high_weight = curriculum_cfg.get(f"{modality_key}_high_weight", shared_high_weight)
    modality_low_weight = curriculum_cfg.get(f"{modality_key}_low_weight", shared_low_weight)

    high_weight = float(modality_high_weight if modality_high_weight is not None else default_weight)
    low_weight = float(modality_low_weight if modality_low_weight is not None else default_weight)
    if float(epoch_value) <= float(high_epochs):
        return high_weight
    return low_weight


def append_metrics_history(history_path: Path, record: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def plot_loss_curve(
    *,
    history_records: list[dict],
    save_path: Path,
) -> Path | None:
    train_records = [
        record
        for record in history_records
        if record.get("split") == "train" and record.get("scope") == "epoch"
    ]
    val_records = [
        record
        for record in history_records
        if record.get("split") == "val"
    ]
    if not train_records and not val_records:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if train_records:
        train_epochs = [int(record["epoch"]) for record in train_records]
        train_losses = [float(record["loss"]) for record in train_records]
        ax.plot(
            train_epochs,
            train_losses,
            marker="o",
            linewidth=2.0,
            label="Train Loss",
        )

    if val_records:
        val_epochs = [float(record["epoch"]) for record in val_records]
        val_losses = [float(record["loss"]) for record in val_records]
        val_labels = [
            "Val Loss (Final)" if record.get("scope") == "final" else "Val Loss"
            for record in val_records
        ]
        plotted_val = False
        for epoch, loss, label in zip(val_epochs, val_losses, val_labels):
            ax.scatter(
                [epoch],
                [loss],
                s=60,
                marker="s" if label.endswith("(Final)") else "o",
                label=label if not plotted_val or label.endswith("(Final)") else None,
                zorder=3,
            )
            plotted_val = True
        if len(val_epochs) > 1:
            ax.plot(
                val_epochs,
                val_losses,
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
                color="C1",
            )

    ax.set_title("Hybrid Diffusion Train / Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def load_target_frequencies_from_dataset(
    dataset: GenRecTokenizedDataset,
    *,
    min_items: int | None = None,
) -> np.ndarray:
    target_item_ids = np.asarray(dataset.arrays["target_item_ids"], dtype=np.int64)
    valid_target_item_ids = target_item_ids[target_item_ids >= 0]
    minlength = dataset.max_item_id + 1
    if min_items is not None:
        minlength = max(int(min_items), minlength)
    return np.bincount(valid_target_item_ids, minlength=minlength).astype(np.int64, copy=False)


def build_inverse_frequency_weights(
    frequencies: np.ndarray,
    *,
    power: float,
    min_weight: float,
    max_weight: float,
    offset: float = 1.0,
) -> np.ndarray:
    freq = np.asarray(frequencies, dtype=np.float64)
    weights = np.power(freq + float(offset), -float(power))
    valid_mask = freq > 0
    if valid_mask.any():
        weights = weights / max(float(weights[valid_mask].mean()), 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))
    return weights.astype(np.float32, copy=False)


def summarize_weight_vector(weights: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(weights)),
        "mean": float(np.mean(weights)),
        "max": float(np.max(weights)),
    }


def build_popularity_bucket_ids(
    frequencies: np.ndarray,
    *,
    num_buckets: int,
) -> np.ndarray:
    if num_buckets <= 1:
        return np.zeros_like(frequencies, dtype=np.int64)
    log_freq = np.log1p(np.asarray(frequencies, dtype=np.float64))
    valid = log_freq[frequencies > 0]
    if valid.size == 0:
        return np.zeros_like(frequencies, dtype=np.int64)
    quantiles = np.linspace(0.0, 1.0, num_buckets + 1, dtype=np.float64)[1:-1]
    boundaries = np.quantile(valid, quantiles)
    bucket_ids = np.digitize(log_freq, boundaries, right=True)
    return bucket_ids.astype(np.int64, copy=False)


def prepare_target_latents(
    *,
    batch: dict[str, torch.Tensor],
    item_latent_table: torch.Tensor,
    normalize: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    if "target_item_latent" in batch:
        latents = batch["target_item_latent"].to(device=item_latent_table.device, dtype=dtype)
    else:
        target_item_ids = batch["target_item_ids"].to(device=item_latent_table.device, dtype=torch.long)
        latents = item_latent_table[target_item_ids].to(dtype=dtype)
    if normalize:
        latents = F.normalize(latents.float(), dim=-1).to(dtype=dtype)
    return latents


@torch.no_grad()
def evaluate(
    *,
    accelerator: Accelerator,
    model: GenRecHybridDiffusionRunner,
    dataloader: DataLoader | None,
    item_latent_table: torch.Tensor,
    normalize_target_latent: bool,
    diffusion_loss_weight: float,
    ranking_loss_weight: float,
    ranking_temperature: float,
    text_auxiliary_loss_weight: float,
    image_auxiliary_loss_weight: float,
    text_history_auxiliary_loss_weight: float,
    image_history_auxiliary_loss_weight: float,
    auxiliary_ranking_temperature: float | None,
    desc: str,
) -> dict[str, float]:
    if dataloader is None:
        return {}

    base_model = accelerator.unwrap_model(model)
    model.eval()
    diffusion_losses: list[float] = []
    ranking_losses: list[float] = []
    text_auxiliary_losses: list[float] = []
    image_auxiliary_losses: list[float] = []
    text_history_auxiliary_losses: list[float] = []
    image_history_auxiliary_losses: list[float] = []
    total_losses: list[float] = []
    progress_bar = tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        desc=desc,
    )
    for batch in progress_bar:
        target_latents = prepare_target_latents(
            batch=batch,
            item_latent_table=item_latent_table,
            normalize=normalize_target_latent,
            dtype=base_model.token_embed.weight.dtype,
        )
        noisy_target_latents, noise, timesteps = base_model.prepare_training_inputs(target_latents=target_latents)

        outputs = model(
            **batch,
            noisy_target_latents=noisy_target_latents,
            timesteps=timesteps,
        )
        loss_dict = base_model.compute_losses(
            output=outputs,
            target_latents=target_latents,
            noise=noise,
            diffusion_loss_weight=diffusion_loss_weight,
            ranking_loss_weight=ranking_loss_weight,
            ranking_temperature=ranking_temperature,
            text_auxiliary_loss_weight=text_auxiliary_loss_weight,
            image_auxiliary_loss_weight=image_auxiliary_loss_weight,
            text_history_auxiliary_loss_weight=text_history_auxiliary_loss_weight,
            image_history_auxiliary_loss_weight=image_history_auxiliary_loss_weight,
            auxiliary_ranking_temperature=auxiliary_ranking_temperature,
            target_item_ids=batch.get("target_item_ids"),
            item_embedding_table=item_latent_table,
        )
        gathered_total = accelerator.gather_for_metrics(loss_dict["loss"].detach().reshape(1))
        gathered_diff = accelerator.gather_for_metrics(loss_dict["diffusion_loss"].detach().reshape(1))
        total_losses.append(float(gathered_total.mean().item()))
        diffusion_losses.append(float(gathered_diff.mean().item()))
        if "ranking_loss" in loss_dict:
            gathered_rank = accelerator.gather_for_metrics(loss_dict["ranking_loss"].detach().reshape(1))
            ranking_losses.append(float(gathered_rank.mean().item()))
        if "text_auxiliary_loss" in loss_dict:
            gathered_text_aux = accelerator.gather_for_metrics(loss_dict["text_auxiliary_loss"].detach().reshape(1))
            text_auxiliary_losses.append(float(gathered_text_aux.mean().item()))
        if "image_auxiliary_loss" in loss_dict:
            gathered_image_aux = accelerator.gather_for_metrics(loss_dict["image_auxiliary_loss"].detach().reshape(1))
            image_auxiliary_losses.append(float(gathered_image_aux.mean().item()))
        if "text_history_auxiliary_loss" in loss_dict:
            gathered_text_history_aux = accelerator.gather_for_metrics(
                loss_dict["text_history_auxiliary_loss"].detach().reshape(1)
            )
            text_history_auxiliary_losses.append(float(gathered_text_history_aux.mean().item()))
        if "image_history_auxiliary_loss" in loss_dict:
            gathered_image_history_aux = accelerator.gather_for_metrics(
                loss_dict["image_history_auxiliary_loss"].detach().reshape(1)
            )
            image_history_auxiliary_losses.append(float(gathered_image_history_aux.mean().item()))

    model.train()
    metrics = {}
    if total_losses:
        metrics["loss"] = float(sum(total_losses) / len(total_losses))
    if diffusion_losses:
        metrics["diffusion_loss"] = float(sum(diffusion_losses) / len(diffusion_losses))
    if ranking_losses:
        metrics["ranking_loss"] = float(sum(ranking_losses) / len(ranking_losses))
    if text_auxiliary_losses:
        metrics["text_auxiliary_loss"] = float(sum(text_auxiliary_losses) / len(text_auxiliary_losses))
    if image_auxiliary_losses:
        metrics["image_auxiliary_loss"] = float(sum(image_auxiliary_losses) / len(image_auxiliary_losses))
    if text_history_auxiliary_losses:
        metrics["text_history_auxiliary_loss"] = float(
            sum(text_history_auxiliary_losses) / len(text_history_auxiliary_losses)
        )
    if image_history_auxiliary_losses:
        metrics["image_history_auxiliary_loss"] = float(
            sum(image_history_auxiliary_losses) / len(image_history_auxiliary_losses)
        )
    return metrics


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: GenRecHybridDiffusionRunner,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    output_dir: Path,
    step: int,
    run_manifest: dict,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_state_dict = accelerator.unwrap_model(model).state_dict()
    torch.save(model_state_dict, checkpoint_dir / "pytorch_model.bin")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
    with open(checkpoint_dir / "training_state.json", "w", encoding="utf-8") as fp:
        json.dump({"global_step": int(step)}, fp, indent=2, ensure_ascii=False)
    with open(checkpoint_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(run_manifest, fp, indent=2, ensure_ascii=False)


def save_best_snapshot(
    *,
    accelerator: Accelerator,
    model: GenRecHybridDiffusionRunner,
    output_dir: Path,
    run_manifest: dict,
    step: int,
    epoch: float,
    metrics: dict[str, float],
) -> Path:
    accelerator.wait_for_everyone()
    best_dir = output_dir / "best"
    if accelerator.is_main_process:
        best_dir.mkdir(parents=True, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), best_dir / "pytorch_model.bin")
        payload = dict(run_manifest)
        payload["best_step"] = int(step)
        payload["best_epoch"] = float(epoch)
        payload["best_metrics"] = {key: float(value) for key, value in metrics.items()}
        with open(best_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()
    return best_dir


def is_metric_improved(
    *,
    current: float,
    best: float | None,
    mode: str,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    return current < (best - min_delta)


def infer_step_from_checkpoint_dir(path: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)", path.name)
    if match:
        return int(match.group(1))
    return None


def resume_from_checkpoint(
    *,
    accelerator: Accelerator,
    model: GenRecHybridDiffusionRunner,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    checkpoint_dir: Path,
) -> int:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    model_path = checkpoint_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    maybe_print(accelerator, f"[resume] loading model from {model_path}")
    model_state = torch.load(model_path, map_location="cpu")
    missing_keys, unexpected_keys = accelerator.unwrap_model(model).load_state_dict(
        model_state,
        strict=False,
    )
    if missing_keys:
        maybe_print(accelerator, f"[resume] missing keys (first 10): {missing_keys[:10]}")
    if unexpected_keys:
        maybe_print(accelerator, f"[resume] unexpected keys (first 10): {unexpected_keys[:10]}")

    step = None
    training_state_path = checkpoint_dir / "training_state.json"
    if training_state_path.exists():
        with open(training_state_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        step = int(payload.get("global_step", 0))
    if step is None:
        step = infer_step_from_checkpoint_dir(checkpoint_dir)
    if step is None:
        step = 0

    optimizer_path = checkpoint_dir / "optimizer.pt"
    optimizer_restored = False
    if optimizer_path.exists():
        maybe_print(accelerator, f"[resume] loading optimizer state from {optimizer_path}")
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        try:
            optimizer.load_state_dict(optimizer_state)
            optimizer_restored = True
        except ValueError as exc:
            maybe_print(
                accelerator,
                f"[resume] optimizer state is incompatible with the current model; "
                f"optimizer will be reset. reason={exc}",
            )
    else:
        maybe_print(
            accelerator,
            f"[resume] optimizer state missing at {optimizer_path}; optimizer will be reset.",
        )

    scheduler_path = checkpoint_dir / "scheduler.pt"
    if scheduler_path.exists() and optimizer_restored:
        maybe_print(accelerator, f"[resume] loading scheduler state from {scheduler_path}")
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
        lr_scheduler.load_state_dict(scheduler_state)
    else:
        maybe_print(
            accelerator,
            f"[resume] scheduler state missing/incompatible at {scheduler_path}; "
            f"trying to fast-forward scheduler to step={step}.",
        )
        if step > 0:
            try:
                lr_scheduler.step(step)
            except TypeError:
                for _ in range(step):
                    lr_scheduler.step()
    maybe_print(accelerator, f"[resume] restored global_step={step}")
    return step


def load_model_only_from_checkpoint(
    *,
    accelerator: Accelerator,
    model: GenRecHybridDiffusionRunner,
    checkpoint_dir: Path,
) -> None:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
    model_path = checkpoint_dir / "pytorch_model.bin"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    maybe_print(accelerator, f"[warmstart] loading model-only weights from {model_path}")
    model_state = torch.load(model_path, map_location="cpu")
    missing_keys, unexpected_keys = accelerator.unwrap_model(model).load_state_dict(
        model_state,
        strict=False,
    )
    if missing_keys:
        maybe_print(accelerator, f"[warmstart] missing keys (first 10): {missing_keys[:10]}")
    if unexpected_keys:
        maybe_print(accelerator, f"[warmstart] unexpected keys (first 10): {unexpected_keys[:10]}")


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_path)

    experiment_name = config.get("experiment_name", "genrec_hybrid_diffusion")
    data_cfg = config.get("data", {})
    backbone_cfg = config.get("backbone", {})
    conditioning_cfg = config.get("conditioning", {})
    representation_cfg = config.get("representation", {})
    diffusion_cfg = config.get("diffusion", {})
    training_cfg = config.get("training", {})

    output_dir = args.output_dir or Path("checkpoints") / experiment_name
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(args.seed)

    train_split = data_cfg.get("train_split", "train")
    valid_split = data_cfg.get("valid_split", "val")
    tokenized_root = Path(data_cfg["tokenized_root"])

    target_latent_path = (
        representation_cfg.get("target_latent_path")
        or representation_cfg.get("embedding_source")
    )
    if not target_latent_path:
        raise ValueError(
            "Config must define `representation.target_latent_path` (or fallback `representation.embedding_source`)."
        )
    item_latent_table = load_item_latent_table(target_latent_path)
    latent_dim = int(representation_cfg.get("latent_dim", item_latent_table.shape[1]))
    if item_latent_table.shape[1] != latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: table has {item_latent_table.shape[1]} dims but config sets {latent_dim}."
        )
    normalize_target_latent = bool(representation_cfg.get("normalize_target_latent", True))

    text_embedding_path = conditioning_cfg.get("text_embedding_path") if conditioning_cfg.get("use_text", False) else None
    use_image = bool(
        conditioning_cfg.get("use_image", False)
        or conditioning_cfg.get("use_history_images", False)
        or conditioning_cfg.get("use_target_image", False)
    )
    image_embedding_path = conditioning_cfg.get("image_embedding_path") if use_image else None
    cf_embedding_path = conditioning_cfg.get("cf_embedding_path") if conditioning_cfg.get("use_cf", False) else None

    train_dataset = GenRecTokenizedDataset(
        tokenized_root=tokenized_root,
        split=train_split,
        text_embedding_path=text_embedding_path,
        image_embedding_path=image_embedding_path,
        cf_embedding_path=cf_embedding_path,
        target_latent_path=target_latent_path,
    )

    valid_dataset = None
    valid_split_root = tokenized_root / valid_split / "samples.npz"
    if valid_split_root.exists():
        valid_dataset = GenRecTokenizedDataset(
            tokenized_root=tokenized_root,
            split=valid_split,
            text_embedding_path=text_embedding_path,
            image_embedding_path=image_embedding_path,
            cf_embedding_path=cf_embedding_path,
            target_latent_path=target_latent_path,
        )

    train_batch_size = int(resolve_training_value(args.train_batch_size, training_cfg, "batch_size", 16))
    eval_batch_size = int(args.eval_batch_size or train_batch_size)
    num_train_epochs = int(resolve_training_value(args.num_train_epochs, training_cfg, "num_train_epochs", 1))
    max_train_steps = resolve_training_value(args.max_train_steps, training_cfg, "max_train_steps", None)
    learning_rate = float(resolve_training_value(args.learning_rate, training_cfg, "learning_rate", 1e-4))
    weight_decay = float(resolve_training_value(args.weight_decay, training_cfg, "weight_decay", 0.01))
    lr_scheduler_name = str(resolve_training_value(args.lr_scheduler, training_cfg, "lr_scheduler", "linear"))
    warmup_steps = int(resolve_training_value(args.warmup_steps, training_cfg, "warmup_steps", 0))
    eval_steps = resolve_training_value(args.eval_steps, training_cfg, "eval_steps", None)
    save_steps = resolve_training_value(args.save_steps, training_cfg, "save_steps", None)
    eval_every_epochs = resolve_training_value(
        args.eval_every_epochs,
        training_cfg,
        "eval_every_epochs",
        None,
    )
    save_every_epochs = resolve_training_value(
        args.save_every_epochs,
        training_cfg,
        "save_every_epochs",
        None,
    )
    diffusion_loss_weight = float(training_cfg.get("diffusion_loss_weight", 1.0))
    ranking_loss_weight = float(training_cfg.get("ranking_loss_weight", 0.0))
    ranking_temperature = float(training_cfg.get("ranking_temperature", 0.07))
    auxiliary_retrieval_cfg = training_cfg.get("auxiliary_retrieval", {})
    auxiliary_retrieval_enabled = bool(auxiliary_retrieval_cfg.get("enabled", False))
    text_auxiliary_loss_weight = float(
        auxiliary_retrieval_cfg.get("text_weight", 0.0 if not auxiliary_retrieval_enabled else 0.1)
    )
    image_auxiliary_loss_weight = float(
        auxiliary_retrieval_cfg.get("image_weight", 0.0 if not auxiliary_retrieval_enabled else 0.1)
    )
    text_history_auxiliary_loss_weight = float(
        auxiliary_retrieval_cfg.get("text_history_weight", 0.0)
    )
    image_history_auxiliary_loss_weight = float(
        auxiliary_retrieval_cfg.get("image_history_weight", 0.0)
    )
    auxiliary_ranking_temperature_cfg = auxiliary_retrieval_cfg.get("temperature")
    auxiliary_ranking_temperature = (
        float(auxiliary_ranking_temperature_cfg)
        if auxiliary_ranking_temperature_cfg is not None
        else None
    )
    auxiliary_retrieval_curriculum_cfg = auxiliary_retrieval_cfg.get("curriculum", {})
    tail_sampler_cfg = training_cfg.get("tail_sampler", {})
    tail_sampler_enabled = bool(tail_sampler_cfg.get("enabled", False))
    tail_sampler_power = float(tail_sampler_cfg.get("power", 0.5))
    tail_sampler_min_weight = float(tail_sampler_cfg.get("min_weight", 0.25))
    tail_sampler_max_weight = float(tail_sampler_cfg.get("max_weight", 4.0))
    ranking_reweight_cfg = training_cfg.get("ranking_reweight", {})
    ranking_reweight_enabled = bool(ranking_reweight_cfg.get("enabled", False))
    ranking_reweight_power = float(ranking_reweight_cfg.get("power", 0.5))
    ranking_reweight_min_weight = float(ranking_reweight_cfg.get("min_weight", 0.5))
    ranking_reweight_max_weight = float(ranking_reweight_cfg.get("max_weight", 3.0))
    val_subset_size_cfg = training_cfg.get("val_subset_size")
    val_subset_size = int(val_subset_size_cfg) if val_subset_size_cfg not in (None, 0) else None
    val_subset_offset = int(training_cfg.get("val_subset_offset", 0))
    run_train_final_eval = bool(training_cfg.get("run_train_final_eval", True))
    early_stopping_cfg = training_cfg.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_metric = str(early_stopping_cfg.get("metric", "loss"))
    early_stopping_mode = str(early_stopping_cfg.get("mode", "min")).lower()
    if early_stopping_mode not in {"min", "max"}:
        raise ValueError(f"Unsupported early stopping mode: {early_stopping_mode}")
    early_stopping_patience = int(early_stopping_cfg.get("patience", 3))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_warmup_epochs = int(early_stopping_cfg.get("warmup_epochs", 0))
    save_best_snapshot_enabled = bool(early_stopping_cfg.get("save_best_snapshot", True))
    use_popularity = bool(conditioning_cfg.get("use_popularity", False))
    popularity_num_buckets = int(conditioning_cfg.get("popularity_num_buckets", 16))
    popularity_cond_dim = int(conditioning_cfg.get("popularity_cond_dim", 32))
    use_popularity_history = bool(conditioning_cfg.get("use_popularity_history", use_popularity))
    use_popularity_pooled = bool(conditioning_cfg.get("use_popularity_pooled", use_popularity))
    conditioning_strategy_kwargs = resolve_hybrid_conditioning_strategy(conditioning_cfg)

    train_item_frequencies = load_target_frequencies_from_dataset(
        train_dataset,
        min_items=item_latent_table.shape[0],
    )
    popularity_bucket_ids = None
    if use_popularity:
        popularity_bucket_ids = torch.from_numpy(
            build_popularity_bucket_ids(
                train_item_frequencies,
                num_buckets=popularity_num_buckets,
            )
        )
    train_item_weights = build_inverse_frequency_weights(
        train_item_frequencies,
        power=ranking_reweight_power,
        min_weight=ranking_reweight_min_weight,
        max_weight=ranking_reweight_max_weight,
    )
    ranking_item_weights = None
    if ranking_reweight_enabled and ranking_loss_weight > 0:
        ranking_item_weights = torch.from_numpy(train_item_weights.copy())

    train_sampler = None
    tail_sample_weights_summary = None
    if tail_sampler_enabled:
        tail_sampler_item_weights = build_inverse_frequency_weights(
            train_item_frequencies,
            power=tail_sampler_power,
            min_weight=tail_sampler_min_weight,
            max_weight=tail_sampler_max_weight,
        )
        train_target_item_ids = np.asarray(train_dataset.arrays["target_item_ids"], dtype=np.int64)
        sample_weights = tail_sampler_item_weights[train_target_item_ids]
        tail_sample_weights_summary = summarize_weight_vector(sample_weights)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=int(sample_weights.shape[0]),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_dataset = maybe_build_fixed_eval_subset(
            valid_dataset,
            subset_size=val_subset_size,
            subset_offset=val_subset_offset,
        )
        valid_loader = make_dataloader(
            valid_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    tokenized_manifest = train_dataset.manifest
    history_len = int(tokenized_manifest["splits"][train_split]["history_len"])
    text_cond_dim = infer_embedding_dim(text_embedding_path)
    image_cond_dim = infer_embedding_dim(image_embedding_path)
    cf_cond_dim = infer_embedding_dim(cf_embedding_path)

    model = GenRecHybridDiffusionRunner.from_batch_metadata(
        manifest=tokenized_manifest,
        max_history_len=history_len,
        latent_dim=latent_dim,
        prediction_type=str(diffusion_cfg.get("prediction_type", "epsilon")),
        num_train_timesteps=int(diffusion_cfg.get("num_train_timesteps", 1000)),
        beta_schedule=str(diffusion_cfg.get("beta_schedule", "squaredcos_cap_v2")),
        hidden_size=int(backbone_cfg.get("hidden_size", 1024)),
        depth=int(backbone_cfg.get("depth", 12)),
        num_heads=int(backbone_cfg.get("num_heads", 16)),
        text_cond_dim=text_cond_dim,
        image_cond_dim=image_cond_dim,
        cf_cond_dim=cf_cond_dim,
        popularity_cond_dim=(popularity_cond_dim if use_popularity else None),
        popularity_num_buckets=(popularity_num_buckets if use_popularity else None),
        item_popularity_bucket_ids=popularity_bucket_ids,
        use_text_history=bool(conditioning_cfg.get("use_text_history", conditioning_cfg.get("use_text", False))),
        use_text_pooled=bool(conditioning_cfg.get("use_text_pooled", conditioning_cfg.get("use_text", False))),
        use_text_target=bool(conditioning_cfg.get("use_target_text", False)),
        use_image_history=bool(conditioning_cfg.get("use_image_history", conditioning_cfg.get("use_history_images", use_image))),
        use_image_pooled=bool(conditioning_cfg.get("use_image_pooled", use_image)),
        use_image_target=bool(conditioning_cfg.get("use_target_image", False)),
        use_cf_history=bool(conditioning_cfg.get("use_cf_history", conditioning_cfg.get("use_cf", False))),
        use_cf_pooled=bool(conditioning_cfg.get("use_cf_pooled", conditioning_cfg.get("use_cf", False))),
        use_cf_target=bool(conditioning_cfg.get("use_target_cf", False)),
        use_popularity_history=use_popularity_history,
        use_popularity_pooled=use_popularity_pooled,
        **conditioning_strategy_kwargs,
        dropout=float(backbone_cfg.get("dropout", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if valid_loader is None:
        model, optimizer, train_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
        )
    else:
        model, optimizer, train_loader, valid_loader = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            valid_loader,
        )

    # Important: compute effective steps after `accelerator.prepare(...)`,
    # because distributed dataloader length may differ from the raw loader.
    num_update_steps_per_epoch = max(1, math.ceil(len(train_loader)))
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    else:
        max_train_steps = int(max_train_steps)
        num_train_epochs = max(1, math.ceil(max_train_steps / num_update_steps_per_epoch))

    lr_scheduler = get_scheduler(
        name=lr_scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    item_latent_table = item_latent_table.to(device=accelerator.device, dtype=torch.float32)
    if ranking_item_weights is not None:
        ranking_item_weights = ranking_item_weights.to(device=accelerator.device, dtype=torch.float32)
    base_model = accelerator.unwrap_model(model)

    run_manifest = {
        "experiment_name": experiment_name,
        "config_path": str(args.config_path),
        "output_dir": str(output_dir),
        "resume_from_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
        "load_model_only_from_checkpoint": (
            str(args.load_model_only_from_checkpoint)
            if args.load_model_only_from_checkpoint
            else None
        ),
        "tokenized_root": str(tokenized_root),
        "train_split": train_split,
        "valid_split": valid_split if valid_dataset is not None else None,
        "train_num_samples": int(len(train_dataset)),
        "valid_num_samples": int(len(valid_dataset)) if valid_dataset is not None else None,
        "target_latent_path": str(target_latent_path),
        "latent_dim": latent_dim,
        "normalize_target_latent": normalize_target_latent,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_scheduler": lr_scheduler_name,
        "warmup_steps": warmup_steps,
        "max_train_steps": max_train_steps,
        "num_train_epochs": num_train_epochs,
        "history_len": history_len,
        "text_embedding_path": text_embedding_path,
        "image_embedding_path": image_embedding_path,
        "cf_embedding_path": cf_embedding_path,
        "use_popularity": use_popularity,
        "popularity_num_buckets": popularity_num_buckets if use_popularity else None,
        "popularity_cond_dim": popularity_cond_dim if use_popularity else None,
        "conditioning_strategy": conditioning_strategy_kwargs,
        "text_cond_dim": text_cond_dim,
        "image_cond_dim": image_cond_dim,
        "cf_cond_dim": cf_cond_dim,
        "mixed_precision": args.mixed_precision,
        "prediction_type": base_model.prediction_type,
        "num_train_timesteps": int(base_model.noise_scheduler.config.num_train_timesteps),
        "beta_schedule": str(base_model.noise_scheduler.config.beta_schedule),
        "diffusion_loss_weight": diffusion_loss_weight,
        "ranking_loss_weight": ranking_loss_weight,
        "ranking_temperature": ranking_temperature,
        "auxiliary_retrieval": {
            "enabled": auxiliary_retrieval_enabled,
            "text_weight": text_auxiliary_loss_weight,
            "image_weight": image_auxiliary_loss_weight,
            "text_history_weight": text_history_auxiliary_loss_weight,
            "image_history_weight": image_history_auxiliary_loss_weight,
            "temperature": auxiliary_ranking_temperature,
            "curriculum": auxiliary_retrieval_curriculum_cfg,
        },
        "eval_steps": int(eval_steps) if eval_steps else None,
        "save_steps": int(save_steps) if save_steps else None,
        "eval_every_epochs": int(eval_every_epochs) if eval_every_epochs else None,
        "save_every_epochs": int(save_every_epochs) if save_every_epochs else None,
        "val_subset_size": int(val_subset_size) if val_subset_size is not None else None,
        "val_subset_offset": int(val_subset_offset),
        "run_train_final_eval": run_train_final_eval,
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "metric": early_stopping_metric,
            "mode": early_stopping_mode,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
            "warmup_epochs": early_stopping_warmup_epochs,
            "save_best_snapshot": save_best_snapshot_enabled,
        },
        "tail_sampler": {
            "enabled": tail_sampler_enabled,
            "power": tail_sampler_power,
            "min_weight": tail_sampler_min_weight,
            "max_weight": tail_sampler_max_weight,
            "sample_weight_summary": tail_sample_weights_summary,
        },
        "ranking_reweight": {
            "enabled": ranking_reweight_enabled and ranking_loss_weight > 0,
            "power": ranking_reweight_power,
            "min_weight": ranking_reweight_min_weight,
            "max_weight": ranking_reweight_max_weight,
            "item_weight_summary": summarize_weight_vector(train_item_weights),
        },
    }

    maybe_print(accelerator, "========== Hybrid GenRec Diffusion Training ==========")
    maybe_print(accelerator, json.dumps(run_manifest, indent=2, ensure_ascii=False))

    metrics_history_path = output_dir / "metrics_history.jsonl"
    loss_curve_path = output_dir / "loss_curve.png"
    if accelerator.is_main_process and metrics_history_path.exists():
        metrics_history_path.unlink()
    accelerator.wait_for_everyone()
    metrics_history_records: list[dict] = []

    if early_stopping_enabled and valid_loader is None:
        maybe_print(
            accelerator,
            "[warn] early stopping is enabled but no validation split was found; disabling early stopping.",
        )
        early_stopping_enabled = False

    best_val_metric: float | None = None
    best_val_metrics: dict[str, float] = {}
    best_val_step: int | None = None
    best_val_epoch: float | None = None
    no_improvement_evals = 0
    stop_training_early = False
    early_stop_reason: str | None = None

    global_step = 0
    if args.load_model_only_from_checkpoint is not None:
        load_model_only_from_checkpoint(
            accelerator=accelerator,
            model=model,
            checkpoint_dir=args.load_model_only_from_checkpoint,
        )
    elif args.resume_from_checkpoint is not None:
        global_step = resume_from_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_dir=args.resume_from_checkpoint,
        )

    def handle_validation_result(metrics: dict[str, float], *, epoch_value: float, scope: str) -> None:
        nonlocal best_val_metric, best_val_metrics, best_val_step, best_val_epoch
        nonlocal no_improvement_evals, stop_training_early, early_stop_reason

        if not early_stopping_enabled:
            return
        if early_stopping_metric not in metrics:
            maybe_print(
                accelerator,
                f"[warn] early stopping metric `{early_stopping_metric}` not found in validation metrics; skipping update.",
            )
            return
        if epoch_value < float(early_stopping_warmup_epochs):
            return

        current_metric = float(metrics[early_stopping_metric])
        improved = is_metric_improved(
            current=current_metric,
            best=best_val_metric,
            mode=early_stopping_mode,
            min_delta=early_stopping_min_delta,
        )
        if improved:
            best_val_metric = current_metric
            best_val_metrics = {key: float(value) for key, value in metrics.items()}
            best_val_step = int(global_step)
            best_val_epoch = float(epoch_value)
            no_improvement_evals = 0
            if save_best_snapshot_enabled:
                save_best_snapshot(
                    accelerator=accelerator,
                    model=model,
                    output_dir=output_dir,
                    run_manifest=run_manifest,
                    step=global_step,
                    epoch=epoch_value,
                    metrics=best_val_metrics,
                )
            maybe_print(
                accelerator,
                f"[early-stop] new best {early_stopping_metric}={current_metric:.6f} "
                f"at step={global_step} epoch={epoch_value:.3f} scope={scope}",
            )
            return

        no_improvement_evals += 1
        maybe_print(
            accelerator,
            f"[early-stop] no improvement on {early_stopping_metric} "
            f"(current={current_metric:.6f}, best={best_val_metric:.6f}) "
            f"count={no_improvement_evals}/{early_stopping_patience}",
        )
        if no_improvement_evals >= early_stopping_patience:
            stop_training_early = True
            early_stop_reason = (
                f"patience reached on {early_stopping_metric} after {no_improvement_evals} validation checks"
            )

    progress_bar = tqdm(
        total=max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
        desc="Hybrid Train",
    )

    if global_step >= max_train_steps:
        maybe_print(
            accelerator,
            f"[resume] global_step ({global_step}) already reached max_train_steps ({max_train_steps}), skipping training loop.",
        )

    starting_epoch = 0
    if num_update_steps_per_epoch > 0 and global_step > 0:
        starting_epoch = min(global_step // num_update_steps_per_epoch, num_train_epochs)

    for epoch_idx in range(starting_epoch, num_train_epochs):
        model.train()
        current_epoch_value = float(epoch_idx + 1)
        current_text_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=text_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=current_epoch_value,
            modality_key="text",
        )
        current_image_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=image_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=current_epoch_value,
            modality_key="image",
        )
        current_text_history_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=text_history_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=current_epoch_value,
            modality_key="text_history",
        )
        current_image_history_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=image_history_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=current_epoch_value,
            modality_key="image_history",
        )
        maybe_print(
            accelerator,
            f"[epoch {epoch_idx + 1}] auxiliary weights: "
            f"text={current_text_auxiliary_loss_weight:.4f} "
            f"image={current_image_auxiliary_loss_weight:.4f} "
            f"text_hist={current_text_history_auxiliary_loss_weight:.4f} "
            f"image_hist={current_image_history_auxiliary_loss_weight:.4f}",
        )
        epoch_train_losses: list[float] = []
        epoch_train_diffusion_losses: list[float] = []
        epoch_train_ranking_losses: list[float] = []
        epoch_train_text_auxiliary_losses: list[float] = []
        epoch_train_image_auxiliary_losses: list[float] = []
        epoch_train_text_history_auxiliary_losses: list[float] = []
        epoch_train_image_history_auxiliary_losses: list[float] = []
        for batch in train_loader:
            if global_step >= max_train_steps:
                break

            target_latents = prepare_target_latents(
                batch=batch,
                item_latent_table=item_latent_table,
                normalize=normalize_target_latent,
                dtype=base_model.token_embed.weight.dtype,
            )
            noisy_target_latents, noise, timesteps = base_model.prepare_training_inputs(
                target_latents=target_latents
            )
            outputs = model(
                **batch,
                noisy_target_latents=noisy_target_latents,
                timesteps=timesteps,
            )
            loss_dict = base_model.compute_losses(
                output=outputs,
                target_latents=target_latents,
                noise=noise,
                diffusion_loss_weight=diffusion_loss_weight,
                ranking_loss_weight=ranking_loss_weight,
                ranking_temperature=ranking_temperature,
                text_auxiliary_loss_weight=current_text_auxiliary_loss_weight,
                image_auxiliary_loss_weight=current_image_auxiliary_loss_weight,
                text_history_auxiliary_loss_weight=current_text_history_auxiliary_loss_weight,
                image_history_auxiliary_loss_weight=current_image_history_auxiliary_loss_weight,
                auxiliary_ranking_temperature=auxiliary_ranking_temperature,
                target_item_ids=batch.get("target_item_ids"),
                item_embedding_table=item_latent_table,
                ranking_sample_weights=(
                    ranking_item_weights[batch["target_item_ids"]]
                    if ranking_item_weights is not None
                    else None
                ),
            )
            loss = loss_dict["loss"]

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{loss.detach().float().item():.4f}",
                diff=f"{loss_dict['diffusion_loss'].detach().float().item():.4f}",
                rank=(
                    f"{loss_dict['ranking_loss'].detach().float().item():.4f}"
                    if "ranking_loss" in loss_dict
                    else "n/a"
                ),
                text_aux=(
                    f"{loss_dict['text_auxiliary_loss'].detach().float().item():.4f}"
                    if "text_auxiliary_loss" in loss_dict
                    else "n/a"
                ),
                text_hist=(
                    f"{loss_dict['text_history_auxiliary_loss'].detach().float().item():.4f}"
                    if "text_history_auxiliary_loss" in loss_dict
                    else "n/a"
                ),
                image_aux=(
                    f"{loss_dict['image_auxiliary_loss'].detach().float().item():.4f}"
                    if "image_auxiliary_loss" in loss_dict
                    else "n/a"
                ),
                image_hist=(
                    f"{loss_dict['image_history_auxiliary_loss'].detach().float().item():.4f}"
                    if "image_history_auxiliary_loss" in loss_dict
                    else "n/a"
                ),
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % max(1, args.logging_steps) == 0:
                msg = (
                    f"[step {global_step}] loss={loss.detach().float().item():.6f} "
                    f"diff={loss_dict['diffusion_loss'].detach().float().item():.6f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                )
                if "ranking_loss" in loss_dict:
                    msg += f" rank={loss_dict['ranking_loss'].detach().float().item():.6f}"
                if "text_auxiliary_loss" in loss_dict:
                    msg += f" text_aux={loss_dict['text_auxiliary_loss'].detach().float().item():.6f}"
                if "text_history_auxiliary_loss" in loss_dict:
                    msg += (
                        f" text_hist_aux={loss_dict['text_history_auxiliary_loss'].detach().float().item():.6f}"
                    )
                if "image_auxiliary_loss" in loss_dict:
                    msg += f" image_aux={loss_dict['image_auxiliary_loss'].detach().float().item():.6f}"
                if "image_history_auxiliary_loss" in loss_dict:
                    msg += (
                        f" image_hist_aux={loss_dict['image_history_auxiliary_loss'].detach().float().item():.6f}"
                    )
                maybe_print(accelerator, msg)

            gathered_total = accelerator.gather_for_metrics(loss.detach().reshape(1))
            gathered_diff = accelerator.gather_for_metrics(loss_dict["diffusion_loss"].detach().reshape(1))
            epoch_train_losses.append(float(gathered_total.mean().item()))
            epoch_train_diffusion_losses.append(float(gathered_diff.mean().item()))
            if "ranking_loss" in loss_dict:
                gathered_rank = accelerator.gather_for_metrics(loss_dict["ranking_loss"].detach().reshape(1))
                epoch_train_ranking_losses.append(float(gathered_rank.mean().item()))
            if "text_auxiliary_loss" in loss_dict:
                gathered_text_aux = accelerator.gather_for_metrics(
                    loss_dict["text_auxiliary_loss"].detach().reshape(1)
                )
                epoch_train_text_auxiliary_losses.append(float(gathered_text_aux.mean().item()))
            if "image_auxiliary_loss" in loss_dict:
                gathered_image_aux = accelerator.gather_for_metrics(
                    loss_dict["image_auxiliary_loss"].detach().reshape(1)
                )
                epoch_train_image_auxiliary_losses.append(float(gathered_image_aux.mean().item()))
            if "text_history_auxiliary_loss" in loss_dict:
                gathered_text_history_aux = accelerator.gather_for_metrics(
                    loss_dict["text_history_auxiliary_loss"].detach().reshape(1)
                )
                epoch_train_text_history_auxiliary_losses.append(float(gathered_text_history_aux.mean().item()))
            if "image_history_auxiliary_loss" in loss_dict:
                gathered_image_history_aux = accelerator.gather_for_metrics(
                    loss_dict["image_history_auxiliary_loss"].detach().reshape(1)
                )
                epoch_train_image_history_auxiliary_losses.append(float(gathered_image_history_aux.mean().item()))

            if eval_steps and valid_loader is not None and global_step % int(eval_steps) == 0:
                metrics = evaluate(
                    accelerator=accelerator,
                    model=model,
                    dataloader=valid_loader,
                    item_latent_table=item_latent_table,
                    normalize_target_latent=normalize_target_latent,
                    diffusion_loss_weight=diffusion_loss_weight,
                    ranking_loss_weight=ranking_loss_weight,
                    ranking_temperature=ranking_temperature,
                    text_auxiliary_loss_weight=current_text_auxiliary_loss_weight,
                    image_auxiliary_loss_weight=current_image_auxiliary_loss_weight,
                    text_history_auxiliary_loss_weight=current_text_history_auxiliary_loss_weight,
                    image_history_auxiliary_loss_weight=current_image_history_auxiliary_loss_weight,
                    auxiliary_ranking_temperature=auxiliary_ranking_temperature,
                    desc=f"Eval@{global_step}",
                )
                if metrics:
                    maybe_print(
                        accelerator,
                        f"[eval {global_step}] "
                        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                    )
                    if accelerator.is_main_process:
                        history_record = {
                            "split": "val",
                            "scope": "step",
                            "step": int(global_step),
                            "epoch": float(epoch_idx + 1),
                            **{key: float(value) for key, value in metrics.items()},
                        }
                        metrics_history_records.append(history_record)
                        append_metrics_history(metrics_history_path, history_record)
                    handle_validation_result(
                        metrics,
                        epoch_value=float(epoch_idx + 1),
                        scope="step",
                    )
                    if stop_training_early:
                        break

            if save_steps and global_step % int(save_steps) == 0:
                save_checkpoint(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    output_dir=output_dir,
                    step=global_step,
                    run_manifest=run_manifest,
                )

            if global_step >= max_train_steps:
                break
            if stop_training_early:
                break

        completed_epoch = epoch_idx + 1
        maybe_print(
            accelerator,
            f"[epoch {completed_epoch}/{num_train_epochs}] completed at global_step={global_step}",
        )

        epoch_metrics = {}
        if epoch_train_losses:
            epoch_metrics["loss"] = float(sum(epoch_train_losses) / len(epoch_train_losses))
        if epoch_train_diffusion_losses:
            epoch_metrics["diffusion_loss"] = float(
                sum(epoch_train_diffusion_losses) / len(epoch_train_diffusion_losses)
            )
        if epoch_train_ranking_losses:
            epoch_metrics["ranking_loss"] = float(
                sum(epoch_train_ranking_losses) / len(epoch_train_ranking_losses)
            )
        if epoch_train_text_auxiliary_losses:
            epoch_metrics["text_auxiliary_loss"] = float(
                sum(epoch_train_text_auxiliary_losses) / len(epoch_train_text_auxiliary_losses)
            )
        if epoch_train_image_auxiliary_losses:
            epoch_metrics["image_auxiliary_loss"] = float(
                sum(epoch_train_image_auxiliary_losses) / len(epoch_train_image_auxiliary_losses)
            )
        if epoch_train_text_history_auxiliary_losses:
            epoch_metrics["text_history_auxiliary_loss"] = float(
                sum(epoch_train_text_history_auxiliary_losses) / len(epoch_train_text_history_auxiliary_losses)
            )
        if epoch_train_image_history_auxiliary_losses:
            epoch_metrics["image_history_auxiliary_loss"] = float(
                sum(epoch_train_image_history_auxiliary_losses) / len(epoch_train_image_history_auxiliary_losses)
            )
        if epoch_metrics:
            maybe_print(
                accelerator,
                f"[train epoch {completed_epoch}] "
                + " ".join(f"{key}={value:.6f}" for key, value in epoch_metrics.items()),
            )
            if accelerator.is_main_process:
                history_record = {
                    "split": "train",
                    "scope": "epoch",
                    "step": int(global_step),
                    "epoch": int(completed_epoch),
                    **epoch_metrics,
                }
                metrics_history_records.append(history_record)
                append_metrics_history(metrics_history_path, history_record)

        if (
            eval_every_epochs
            and valid_loader is not None
            and completed_epoch % int(eval_every_epochs) == 0
        ):
            metrics = evaluate(
                accelerator=accelerator,
                model=model,
                dataloader=valid_loader,
                item_latent_table=item_latent_table,
                normalize_target_latent=normalize_target_latent,
                diffusion_loss_weight=diffusion_loss_weight,
                ranking_loss_weight=ranking_loss_weight,
                ranking_temperature=ranking_temperature,
                text_auxiliary_loss_weight=current_text_auxiliary_loss_weight,
                image_auxiliary_loss_weight=current_image_auxiliary_loss_weight,
                text_history_auxiliary_loss_weight=current_text_history_auxiliary_loss_weight,
                image_history_auxiliary_loss_weight=current_image_history_auxiliary_loss_weight,
                auxiliary_ranking_temperature=auxiliary_ranking_temperature,
                desc=f"EvalEpoch@{completed_epoch}",
            )
            if metrics:
                maybe_print(
                    accelerator,
                    f"[eval epoch {completed_epoch}] "
                    + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                )
                if accelerator.is_main_process:
                    history_record = {
                        "split": "val",
                        "scope": "epoch",
                        "step": int(global_step),
                        "epoch": int(completed_epoch),
                        **{key: float(value) for key, value in metrics.items()},
                    }
                    metrics_history_records.append(history_record)
                    append_metrics_history(metrics_history_path, history_record)
                handle_validation_result(
                    metrics,
                    epoch_value=float(completed_epoch),
                    scope="epoch",
                )

        if save_every_epochs and completed_epoch % int(save_every_epochs) == 0:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                output_dir=output_dir,
                step=global_step,
                run_manifest=run_manifest,
            )

        if global_step >= max_train_steps:
            break
        if stop_training_early:
            maybe_print(
                accelerator,
                f"[early-stop] stopping training after epoch {completed_epoch}: {early_stop_reason}",
            )
            break

    final_metrics = {}
    if valid_loader is not None and run_train_final_eval:
        final_epoch_for_aux = float(completed_epoch if "completed_epoch" in locals() else num_train_epochs)
        final_text_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=text_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=final_epoch_for_aux,
            modality_key="text",
        )
        final_image_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=image_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=final_epoch_for_aux,
            modality_key="image",
        )
        final_text_history_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=text_history_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=final_epoch_for_aux,
            modality_key="text_history",
        )
        final_image_history_auxiliary_loss_weight = resolve_auxiliary_weight_for_epoch(
            default_weight=image_history_auxiliary_loss_weight,
            curriculum_cfg=auxiliary_retrieval_curriculum_cfg,
            epoch_value=final_epoch_for_aux,
            modality_key="image_history",
        )
        final_metrics = evaluate(
            accelerator=accelerator,
            model=model,
            dataloader=valid_loader,
            item_latent_table=item_latent_table,
            normalize_target_latent=normalize_target_latent,
            diffusion_loss_weight=diffusion_loss_weight,
            ranking_loss_weight=ranking_loss_weight,
            ranking_temperature=ranking_temperature,
            text_auxiliary_loss_weight=final_text_auxiliary_loss_weight,
            image_auxiliary_loss_weight=final_image_auxiliary_loss_weight,
            text_history_auxiliary_loss_weight=final_text_history_auxiliary_loss_weight,
            image_history_auxiliary_loss_weight=final_image_history_auxiliary_loss_weight,
            auxiliary_ranking_temperature=auxiliary_ranking_temperature,
            desc="Final Eval",
        )
        if final_metrics:
            maybe_print(
                accelerator,
                "[final eval] " + " ".join(f"{key}={value:.6f}" for key, value in final_metrics.items()),
            )
            if accelerator.is_main_process:
                history_record = {
                    "split": "val",
                    "scope": "final",
                    "step": int(global_step),
                    "epoch": int(num_train_epochs),
                    **{key: float(value) for key, value in final_metrics.items()},
                }
                metrics_history_records.append(history_record)
                append_metrics_history(metrics_history_path, history_record)
    elif valid_loader is not None:
        maybe_print(accelerator, "[final eval] skipped by configuration")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        saved_curve_path = plot_loss_curve(
            history_records=metrics_history_records,
            save_path=loss_curve_path,
        )
        torch.save(accelerator.unwrap_model(model).state_dict(), final_dir / "pytorch_model.bin")
        with open(final_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
            payload = dict(run_manifest)
            payload["final_metrics"] = final_metrics
            payload["best_val_metric"] = best_val_metric
            payload["best_val_metrics"] = best_val_metrics
            payload["best_val_step"] = best_val_step
            payload["best_val_epoch"] = best_val_epoch
            payload["stopped_early"] = stop_training_early
            payload["early_stop_reason"] = early_stop_reason
            payload["metrics_history_path"] = str(metrics_history_path)
            payload["loss_curve_path"] = str(saved_curve_path) if saved_curve_path is not None else None
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"saved_final_model = {final_dir / 'pytorch_model.bin'}")
        print(f"saved_metrics_history = {metrics_history_path}")
        if saved_curve_path is not None:
            print(f"saved_loss_curve = {saved_curve_path}")


if __name__ == "__main__":
    main()
