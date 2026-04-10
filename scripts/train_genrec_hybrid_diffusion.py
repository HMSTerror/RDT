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
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers.optimization import get_scheduler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.data import GenRecTokenizedCollator, GenRecTokenizedDataset  # noqa: E402
from genrec.models import GenRecHybridDiffusionRunner  # noqa: E402


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
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory to resume from (e.g., output_dir/checkpoint-43000).",
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


def maybe_print(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_local_main_process:
        print(message)


def resolve_training_value(cli_value, config_section: dict, key: str, default):
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


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
    desc: str,
) -> dict[str, float]:
    if dataloader is None:
        return {}

    base_model = accelerator.unwrap_model(model)
    model.eval()
    diffusion_losses: list[float] = []
    ranking_losses: list[float] = []
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

    model.train()
    metrics = {}
    if total_losses:
        metrics["loss"] = float(sum(total_losses) / len(total_losses))
    if diffusion_losses:
        metrics["diffusion_loss"] = float(sum(diffusion_losses) / len(diffusion_losses))
    if ranking_losses:
        metrics["ranking_loss"] = float(sum(ranking_losses) / len(ranking_losses))
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
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
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
    diffusion_loss_weight = float(training_cfg.get("diffusion_loss_weight", 1.0))
    ranking_loss_weight = float(training_cfg.get("ranking_loss_weight", 0.0))
    ranking_temperature = float(training_cfg.get("ranking_temperature", 0.07))
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
    use_popularity = bool(conditioning_cfg.get("use_popularity", False))
    popularity_num_buckets = int(conditioning_cfg.get("popularity_num_buckets", 16))
    popularity_cond_dim = int(conditioning_cfg.get("popularity_cond_dim", 32))
    use_popularity_history = bool(conditioning_cfg.get("use_popularity_history", use_popularity))
    use_popularity_pooled = bool(conditioning_cfg.get("use_popularity_pooled", use_popularity))

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
        "tokenized_root": str(tokenized_root),
        "train_split": train_split,
        "valid_split": valid_split if valid_dataset is not None else None,
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

    global_step = 0
    if args.resume_from_checkpoint is not None:
        global_step = resume_from_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_dir=args.resume_from_checkpoint,
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

    for _ in range(num_train_epochs):
        model.train()
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
                maybe_print(accelerator, msg)

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
                    desc=f"Eval@{global_step}",
                )
                if metrics:
                    maybe_print(
                        accelerator,
                        f"[eval {global_step}] "
                        + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()),
                    )

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

        if global_step >= max_train_steps:
            break

    final_metrics = {}
    if valid_loader is not None:
        final_metrics = evaluate(
            accelerator=accelerator,
            model=model,
            dataloader=valid_loader,
            item_latent_table=item_latent_table,
            normalize_target_latent=normalize_target_latent,
            diffusion_loss_weight=diffusion_loss_weight,
            ranking_loss_weight=ranking_loss_weight,
            ranking_temperature=ranking_temperature,
            desc="Final Eval",
        )
        if final_metrics:
            maybe_print(
                accelerator,
                "[final eval] " + " ".join(f"{key}={value:.6f}" for key, value in final_metrics.items()),
            )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        torch.save(accelerator.unwrap_model(model).state_dict(), final_dir / "pytorch_model.bin")
        with open(final_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
            payload = dict(run_manifest)
            payload["final_metrics"] = final_metrics
            json.dump(payload, fp, indent=2, ensure_ascii=False)
        print(f"saved_final_model = {final_dir / 'pytorch_model.bin'}")


if __name__ == "__main__":
    main()
