#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.optimization import get_scheduler


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.data import GenRecTokenizedCollator, GenRecTokenizedDataset  # noqa: E402
from genrec.models import GenRecDiTRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the GenRec DiT-style semantic-ID model on tokenized train/val/test buffers."
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("configs/genrec_dit_amazon.yaml"),
        help="YAML config for the GenRec DiT experiment.",
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


def make_dataloader(
    dataset: GenRecTokenizedDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=GenRecTokenizedCollator(),
    )


def maybe_print(accelerator: Accelerator, message: str) -> None:
    if accelerator.is_local_main_process:
        print(message)


@torch.no_grad()
def evaluate(
    *,
    accelerator: Accelerator,
    model: GenRecDiTRunner,
    dataloader: DataLoader | None,
    desc: str,
) -> dict[str, float]:
    if dataloader is None:
        return {}

    model.eval()
    losses = []
    accuracies = []
    progress_bar = tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        desc=desc,
    )
    for batch in progress_bar:
        outputs = model(**batch)
        if outputs.loss is None:
            continue

        gathered_loss = accelerator.gather_for_metrics(outputs.loss.detach().reshape(1))
        losses.append(float(gathered_loss.mean().item()))

        if outputs.masked_token_accuracy is not None:
            gathered_acc = accelerator.gather_for_metrics(outputs.masked_token_accuracy.detach().reshape(1))
            accuracies.append(float(gathered_acc.mean().item()))

    model.train()
    metrics = {}
    if losses:
        metrics["loss"] = float(sum(losses) / len(losses))
    if accuracies:
        metrics["masked_token_accuracy"] = float(sum(accuracies) / len(accuracies))
    return metrics


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model: GenRecDiTRunner,
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

    state_dict = accelerator.unwrap_model(model).state_dict()
    torch.save(state_dict, checkpoint_dir / "pytorch_model.bin")
    with open(checkpoint_dir / "run_manifest.json", "w", encoding="utf-8") as fp:
        json.dump(run_manifest, fp, indent=2, ensure_ascii=False)


def resolve_training_value(
    cli_value,
    config_section: dict,
    key: str,
    default,
):
    if cli_value is not None:
        return cli_value
    return config_section.get(key, default)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config_path)

    experiment_name = config.get("experiment_name", "genrec_dit")
    data_cfg = config.get("data", {})
    backbone_cfg = config.get("backbone", {})
    conditioning_cfg = config.get("conditioning", {})
    training_cfg = config.get("training", {})

    output_dir = args.output_dir or Path("checkpoints") / experiment_name
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed)

    train_split = data_cfg.get("train_split", "train")
    valid_split = data_cfg.get("valid_split", "val")
    tokenized_root = Path(data_cfg["tokenized_root"])

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

    train_loader = make_dataloader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
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

    model = GenRecDiTRunner.from_batch_metadata(
        manifest=tokenized_manifest,
        max_history_len=history_len,
        hidden_size=int(backbone_cfg.get("hidden_size", 1024)),
        depth=int(backbone_cfg.get("depth", 12)),
        num_heads=int(backbone_cfg.get("num_heads", 16)),
        text_cond_dim=text_cond_dim,
        image_cond_dim=image_cond_dim,
        cf_cond_dim=cf_cond_dim,
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        restrict_target_vocab=bool(training_cfg.get("restrict_target_vocab", True)),
        use_text_history=bool(conditioning_cfg.get("use_text_history", conditioning_cfg.get("use_text", False))),
        use_text_pooled=bool(conditioning_cfg.get("use_text_pooled", conditioning_cfg.get("use_text", False))),
        use_text_target=bool(conditioning_cfg.get("use_target_text", False)),
        use_image_history=bool(conditioning_cfg.get("use_image_history", conditioning_cfg.get("use_history_images", use_image))),
        use_image_pooled=bool(conditioning_cfg.get("use_image_pooled", use_image)),
        use_image_target=bool(conditioning_cfg.get("use_target_image", False)),
        use_cf_history=bool(conditioning_cfg.get("use_cf_history", conditioning_cfg.get("use_cf", False))),
        use_cf_pooled=bool(conditioning_cfg.get("use_cf_pooled", conditioning_cfg.get("use_cf", False))),
        use_cf_target=bool(conditioning_cfg.get("use_target_cf", False)),
        dropout=float(backbone_cfg.get("dropout", 0.0)),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

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

    if valid_loader is None:
        model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            lr_scheduler,
        )
    else:
        model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            valid_loader,
            lr_scheduler,
        )

    run_manifest = {
        "experiment_name": experiment_name,
        "config_path": str(args.config_path),
        "output_dir": str(output_dir),
        "tokenized_root": str(tokenized_root),
        "train_split": train_split,
        "valid_split": valid_split if valid_dataset is not None else None,
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
        "text_cond_dim": text_cond_dim,
        "image_cond_dim": image_cond_dim,
        "cf_cond_dim": cf_cond_dim,
        "mixed_precision": args.mixed_precision,
    }

    maybe_print(accelerator, "========== GenRec DiT Training ==========")
    maybe_print(accelerator, json.dumps(run_manifest, indent=2, ensure_ascii=False))

    global_step = 0
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="GenRec Train",
    )

    for epoch in range(num_train_epochs):
        model.train()
        for batch in train_loader:
            outputs = model(**batch)
            if outputs.loss is None:
                raise RuntimeError("Model did not return a training loss.")

            accelerator.backward(outputs.loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                loss=f"{outputs.loss.detach().float().item():.4f}",
                acc=(
                    f"{outputs.masked_token_accuracy.detach().float().item():.4f}"
                    if outputs.masked_token_accuracy is not None
                    else "n/a"
                ),
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % max(1, args.logging_steps) == 0:
                maybe_print(
                    accelerator,
                    f"[step {global_step}] loss={outputs.loss.detach().float().item():.6f} "
                    f"acc={outputs.masked_token_accuracy.detach().float().item() if outputs.masked_token_accuracy is not None else float('nan'):.6f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.2e}",
                )

            if eval_steps and valid_loader is not None and global_step % int(eval_steps) == 0:
                metrics = evaluate(
                    accelerator=accelerator,
                    model=model,
                    dataloader=valid_loader,
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
            json.dump(
                {
                    **run_manifest,
                    "final_metrics": final_metrics,
                },
                fp,
                indent=2,
                ensure_ascii=False,
            )

    maybe_print(accelerator, f"Training finished at step {global_step}.")


if __name__ == "__main__":
    main()
