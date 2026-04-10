#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.data import GenRecTokenizedCollator, GenRecTokenizedDataset  # noqa: E402
from genrec.models import GenRecHybridDiffusionRunner  # noqa: E402


"""
GROUP_NAMES = ("cold", "mid", "hot")
GROUP_NAMES_ZH = {
    "cold": "冷门",
    "mid": "一般",
    "hot": "热门",
}
"""
GROUP_NAMES = ("cold", "mid", "hot")
GROUP_NAMES_ZH = {
    "cold": "\u51b7\u95e8",
    "mid": "\u4e00\u822c",
    "hot": "\u70ed\u95e8",
}
GROUP_PLOT_LABELS = {
    "cold": "Cold",
    "mid": "Mid",
    "hot": "Hot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Hybrid GenRec Diffusion on train/val/test tokenized buffers, "
            "including popularity-grouped retrieval metrics."
        )
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=Path("configs/genrec_hybrid_diffusion_amazon.yaml"),
        help="YAML config for the hybrid diffusion experiment.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint directory or pytorch_model.bin file saved by train_genrec_hybrid_diffusion.py.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which tokenized split to evaluate.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="DDIM sampling steps during hybrid diffusion inference.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="5,10,20",
        help="Comma-separated top-k list, e.g. '5,10,20'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for evaluation. Defaults to cuda if available, else cpu.",
    )
    parser.add_argument(
        "--exclude_history_items",
        action="store_true",
        help="Exclude valid history items from candidate ranking.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Optional max number of evaluation batches. 0 means full split.",
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="equal_items",
        choices=["equal_items", "quantile_frequency"],
        help=(
            "How to split items into hot/mid/cold groups using train frequency. "
            "`equal_items` sorts items by frequency and splits item ids into 3 equal-sized groups. "
            "`quantile_frequency` uses 33%%/66%% quantiles over frequencies."
        ),
    )
    parser.add_argument(
        "--frequency_source_split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split supplies target-item observation frequency for popularity grouping.",
    )
    parser.add_argument(
        "--popularity_penalty",
        type=float,
        default=None,
        help=(
            "Optional popularity penalty coefficient applied during retrieval reranking. "
            "Uses normalized log(1 + frequency) from the configured source split."
        ),
    )
    parser.add_argument(
        "--popularity_penalty_source_split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Which split supplies item frequencies for popularity reranking. Defaults to the config or frequency_source_split.",
    )
    parser.add_argument(
        "--save_json",
        type=Path,
        default=None,
        help="Optional JSON path for final overall + grouped metrics.",
    )
    parser.add_argument(
        "--save_jsonl",
        type=Path,
        default=None,
        help="Optional JSONL path for per-sample predictions.",
    )
    parser.add_argument(
        "--save_plot",
        type=Path,
        default=None,
        help="Optional PNG path for an automatic hot/mid/cold comparison figure.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=20,
        help="Print running metrics every N batches.",
    )
    parser.add_argument(
        "--occlude_modalities",
        type=str,
        default="",
        help=(
            "Comma-separated modality names to zero out at evaluation time for quick ablation, "
            "e.g. 'text', 'image', 'cf', or 'text,image'."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def parse_topk(topk_arg: str) -> List[int]:
    values = sorted({int(item) for item in topk_arg.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise ValueError("--topk must contain positive integers, e.g. '5,10,20'.")
    return values


def parse_modalities(modality_arg: str) -> List[str]:
    if not modality_arg.strip():
        return []
    values = []
    for item in modality_arg.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in {"text", "image", "cf"}:
            raise ValueError(
                "--occlude_modalities only supports: text, image, cf "
                f"(got {name!r})."
            )
        if name not in values:
            values.append(name)
    return values


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def resolve_checkpoint_file(checkpoint_path: Path) -> Path:
    if checkpoint_path.is_file():
        return checkpoint_path
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = [
        checkpoint_path / "pytorch_model.bin",
        checkpoint_path / "final" / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find pytorch_model.bin inside checkpoint directory: {checkpoint_path}"
    )


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint_file = resolve_checkpoint_file(checkpoint_path)
    payload = torch.load(checkpoint_file, map_location="cpu")
    state_dict = payload["module"] if isinstance(payload, dict) and "module" in payload else payload
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[warn] Missing keys when loading checkpoint: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"[warn] Unexpected keys when loading checkpoint: {unexpected_keys[:10]}")


def build_dataloader(
    *,
    tokenized_root: Path,
    split: str,
    text_embedding_path: str | None,
    image_embedding_path: str | None,
    cf_embedding_path: str | None,
    target_latent_path: str | Path | None,
    batch_size: int,
    num_workers: int,
):
    dataset = GenRecTokenizedDataset(
        tokenized_root=tokenized_root,
        split=split,
        text_embedding_path=text_embedding_path,
        image_embedding_path=image_embedding_path,
        cf_embedding_path=cf_embedding_path,
        target_latent_path=target_latent_path,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=GenRecTokenizedCollator(),
    )
    return dataset, dataloader


def resolve_buffer_split_root(buffer_root: Path, split: str) -> Path:
    candidate = buffer_root / split
    if (candidate / "item_map.json").exists():
        return candidate
    if (buffer_root / "item_map.json").exists():
        return buffer_root
    raise FileNotFoundError(f"Could not resolve split root for `{split}` under {buffer_root}.")


def load_item_index_mapping(
    *,
    buffer_root: Path,
    split_for_item_map: str,
) -> dict[int, str]:
    split_root = resolve_buffer_split_root(buffer_root, split_for_item_map)
    with open(split_root / "item_map.json", "r", encoding="utf-8") as fp:
        item_map = json.load(fp)
    item_map = {str(item_id): int(idx) for item_id, idx in item_map.items()}
    return {idx: item_id for item_id, idx in item_map.items()}


def build_fallback_item_index_mapping(num_items: int) -> dict[int, str]:
    return {idx: f"item_idx_{idx}" for idx in range(int(num_items))}


class RetrievalMetricTracker:
    def __init__(self, topk_list: List[int]) -> None:
        self.topk_list = topk_list
        self.total = 0
        self.sums: Dict[str, float] = {"mean_rank": 0.0}
        for k in topk_list:
            self.sums[f"hit@{k}"] = 0.0
            self.sums[f"recall@{k}"] = 0.0
            self.sums[f"mrr@{k}"] = 0.0
            self.sums[f"ndcg@{k}"] = 0.0

    def update(self, ranks: torch.Tensor) -> None:
        ranks = ranks.detach().cpu().to(torch.long)
        batch_size = int(ranks.shape[0])
        if batch_size == 0:
            return
        self.total += batch_size
        self.sums["mean_rank"] += float(ranks.float().sum().item())
        for k in self.topk_list:
            hit = (ranks <= k).float()
            ndcg = torch.where(
                ranks <= k,
                1.0 / torch.log2(ranks.float() + 1.0),
                torch.zeros_like(ranks, dtype=torch.float32),
            )
            mrr = torch.where(
                ranks <= k,
                1.0 / ranks.float(),
                torch.zeros_like(ranks, dtype=torch.float32),
            )
            self.sums[f"hit@{k}"] += float(hit.sum().item())
            self.sums[f"recall@{k}"] += float(hit.sum().item())
            self.sums[f"mrr@{k}"] += float(mrr.sum().item())
            self.sums[f"ndcg@{k}"] += float(ndcg.sum().item())

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        return {name: value / self.total for name, value in self.sums.items()}


def exclude_history_items_from_scores(
    scores: torch.Tensor,
    history_item_ids: torch.Tensor,
    history_masks: torch.Tensor,
    target_item_ids: torch.Tensor,
) -> torch.Tensor:
    scores = scores.clone()
    target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
    valid_history = history_masks & history_item_ids.ge(0)
    if valid_history.any():
        row_idx = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(history_item_ids)
        col_idx = history_item_ids.clamp_min(0)
        scores[row_idx[valid_history], col_idx[valid_history]] = -torch.inf
        scores.scatter_(1, target_item_ids.unsqueeze(1), target_scores)
    return scores


def load_split_target_frequencies(tokenized_root: Path, split: str, num_items: int) -> np.ndarray:
    samples_path = tokenized_root / split / "samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"Could not find tokenized split samples: {samples_path}")
    with np.load(samples_path, allow_pickle=False) as payload:
        target_item_ids = np.asarray(payload["target_item_ids"], dtype=np.int64)
    target_item_ids = target_item_ids[target_item_ids >= 0]
    return np.bincount(target_item_ids, minlength=num_items).astype(np.int64, copy=False)


def build_popularity_groups(
    frequencies: np.ndarray,
    *,
    strategy: str,
) -> tuple[np.ndarray, dict]:
    num_items = int(frequencies.shape[0])
    item_group_ids = np.zeros(num_items, dtype=np.int64)

    if strategy == "equal_items":
        sorted_items = np.argsort(frequencies, kind="stable")
        cold_idx, mid_idx, hot_idx = np.array_split(sorted_items, 3)
        item_group_ids[cold_idx] = 0
        item_group_ids[mid_idx] = 1
        item_group_ids[hot_idx] = 2
        metadata = {
            "strategy": strategy,
            "split_item_counts": {
                "cold": int(len(cold_idx)),
                "mid": int(len(mid_idx)),
                "hot": int(len(hot_idx)),
            },
        }
    elif strategy == "quantile_frequency":
        q1, q2 = np.quantile(frequencies.astype(np.float64), [1.0 / 3.0, 2.0 / 3.0])
        item_group_ids[frequencies > q1] = 1
        item_group_ids[frequencies > q2] = 2
        metadata = {
            "strategy": strategy,
            "frequency_quantiles": {
                "q33": float(q1),
                "q66": float(q2),
            },
        }
    else:
        raise ValueError(f"Unsupported group strategy: {strategy}")

    group_summaries = {}
    for group_id, group_name in enumerate(GROUP_NAMES):
        mask = item_group_ids == group_id
        group_freqs = frequencies[mask]
        group_summaries[group_name] = {
            "name_zh": GROUP_NAMES_ZH[group_name],
            "num_items": int(mask.sum()),
            "min_frequency": int(group_freqs.min()) if group_freqs.size else 0,
            "max_frequency": int(group_freqs.max()) if group_freqs.size else 0,
            "mean_frequency": float(group_freqs.mean()) if group_freqs.size else 0.0,
        }
    metadata["groups"] = group_summaries
    return item_group_ids, metadata


def build_normalized_popularity_penalty(frequencies: np.ndarray) -> np.ndarray:
    penalties = np.log1p(np.asarray(frequencies, dtype=np.float64))
    max_value = float(np.max(penalties)) if penalties.size else 0.0
    if max_value > 0:
        penalties = penalties / max_value
    return penalties.astype(np.float32, copy=False)


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


def apply_modality_occlusion_inplace(batch: dict[str, torch.Tensor], modalities: List[str]) -> None:
    if not modalities:
        return

    branch_to_keys = {
        "text": (
            "history_text_embeds",
            "target_text_embed",
            "pooled_text_embed",
        ),
        "image": (
            "history_image_embeds",
            "target_image_embed",
            "pooled_image_embed",
        ),
        "cf": (
            "history_cf_embeds",
            "target_cf_embed",
            "pooled_cf_embed",
        ),
    }
    for modality in modalities:
        for key in branch_to_keys[modality]:
            value = batch.get(key)
            if torch.is_tensor(value):
                value.zero_()


def write_jsonl_records(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_plot_path(args: argparse.Namespace) -> Path | None:
    if args.save_plot is not None:
        return args.save_plot
    if args.save_json is not None:
        return args.save_json.with_suffix(".png")
    return None


def plot_grouped_metrics(
    *,
    save_path: Path,
    split: str,
    topk_list: List[int],
    overall_metrics: dict,
    group_metrics: dict,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group_order = list(GROUP_NAMES)
    x = np.arange(len(group_order), dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Hybrid Diffusion Grouped Evaluation on {split} (Cold / Mid / Hot)",
        fontsize=14,
        fontweight="bold",
    )

    mean_rank_values = [float(group_metrics[group_name].get("mean_rank", 0.0)) for group_name in group_order]
    axes[0, 0].bar(
        x,
        mean_rank_values,
        color=["#6c8ebf", "#93c47d", "#e69138"],
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0, 0].set_title("Mean Rank by Group")
    axes[0, 0].set_xticks(x, [GROUP_PLOT_LABELS[group_name] for group_name in group_order])
    axes[0, 0].set_ylabel("Mean Rank")
    axes[0, 0].grid(axis="y", alpha=0.25)
    if "mean_rank" in overall_metrics:
        axes[0, 0].axhline(
            float(overall_metrics["mean_rank"]),
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Overall",
        )
        axes[0, 0].legend(frameon=False)

    metric_panels = [
        ("hit", "Hit@K", axes[0, 1]),
        ("ndcg", "NDCG@K", axes[1, 0]),
        ("mrr", "MRR@K", axes[1, 1]),
    ]
    num_k = len(topk_list)
    bar_width = 0.75 / max(1, num_k)
    offsets = (np.arange(num_k, dtype=np.float32) - (num_k - 1) / 2.0) * bar_width

    for family, title, ax in metric_panels:
        for idx, k in enumerate(topk_list):
            values = [float(group_metrics[group_name].get(f"{family}@{k}", 0.0)) for group_name in group_order]
            label = f"Hit@{k}" if family == "hit" else f"{family.upper()}@{k}"
            ax.bar(
                x + offsets[idx],
                values,
                width=bar_width,
                label=label,
                alpha=0.9,
            )
            overall_key = f"{family}@{k}"
            if overall_key in overall_metrics:
                ax.axhline(
                    float(overall_metrics[overall_key]),
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.35,
                    color=f"C{idx}",
                )

        ax.set_title(title)
        ax.set_xticks(x, [GROUP_PLOT_LABELS[group_name] for group_name in group_order])
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def main() -> None:
    args = parse_args()
    occluded_modalities = parse_modalities(args.occlude_modalities)
    config = load_yaml(args.config_path)
    data_cfg = config.get("data", {})
    backbone_cfg = config.get("backbone", {})
    conditioning_cfg = config.get("conditioning", {})
    representation_cfg = config.get("representation", {})
    diffusion_cfg = config.get("diffusion", {})
    evaluation_cfg = config.get("evaluation", {})

    topk_list = parse_topk(args.topk)
    max_k = max(topk_list)
    device = resolve_device(args.device)

    tokenized_root = Path(data_cfg["tokenized_root"])
    buffer_root = Path(data_cfg["split_root"])
    train_split = data_cfg.get("train_split", "train")

    target_latent_path = (
        representation_cfg.get("target_latent_path")
        or representation_cfg.get("embedding_source")
    )
    if not target_latent_path:
        raise ValueError(
            "Config must define `representation.target_latent_path` (or fallback `representation.embedding_source`)."
        )

    text_embedding_path = conditioning_cfg.get("text_embedding_path") if conditioning_cfg.get("use_text", False) else None
    use_image = bool(
        conditioning_cfg.get("use_image", False)
        or conditioning_cfg.get("use_history_images", False)
        or conditioning_cfg.get("use_target_image", False)
    )
    image_embedding_path = conditioning_cfg.get("image_embedding_path") if use_image else None
    cf_embedding_path = conditioning_cfg.get("cf_embedding_path") if conditioning_cfg.get("use_cf", False) else None
    use_popularity = bool(conditioning_cfg.get("use_popularity", False))
    popularity_num_buckets = int(conditioning_cfg.get("popularity_num_buckets", 16))
    popularity_cond_dim = int(conditioning_cfg.get("popularity_cond_dim", 32))
    use_popularity_history = bool(conditioning_cfg.get("use_popularity_history", use_popularity))
    use_popularity_pooled = bool(conditioning_cfg.get("use_popularity_pooled", use_popularity))

    dataset, dataloader = build_dataloader(
        tokenized_root=tokenized_root,
        split=args.split,
        text_embedding_path=text_embedding_path,
        image_embedding_path=image_embedding_path,
        cf_embedding_path=cf_embedding_path,
        target_latent_path=target_latent_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    tokenized_manifest = dataset.manifest
    history_len = int(tokenized_manifest["splits"][train_split]["history_len"])
    text_cond_dim = infer_embedding_dim(text_embedding_path)
    image_cond_dim = infer_embedding_dim(image_embedding_path)
    cf_cond_dim = infer_embedding_dim(cf_embedding_path)

    item_latent_table = load_item_latent_table(target_latent_path)
    latent_dim = int(representation_cfg.get("latent_dim", item_latent_table.shape[1]))
    if item_latent_table.shape[1] != latent_dim:
        raise ValueError(
            f"Latent dimension mismatch: table has {item_latent_table.shape[1]} dims but config sets {latent_dim}."
        )
    normalize_target_latent = bool(representation_cfg.get("normalize_target_latent", True))

    popularity_bucket_ids = None
    if use_popularity:
        train_item_frequencies = load_split_target_frequencies(
            tokenized_root=tokenized_root,
            split=train_split,
            num_items=item_latent_table.shape[0],
        )
        popularity_bucket_ids = torch.from_numpy(
            build_popularity_bucket_ids(
                train_item_frequencies,
                num_buckets=popularity_num_buckets,
            )
        )

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
    load_model_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    item_latent_table = item_latent_table.to(device=device, dtype=torch.float32)
    if normalize_target_latent:
        item_latent_table = F.normalize(item_latent_table, dim=-1)

    item_id_mapping_source = "buffer_item_map"
    try:
        idx_to_item = load_item_index_mapping(
            buffer_root=buffer_root,
            split_for_item_map=train_split,
        )
        if len(idx_to_item) != item_latent_table.shape[0]:
            raise ValueError(
                f"Item map size ({len(idx_to_item)}) does not match latent table rows ({item_latent_table.shape[0]})."
            )
    except (FileNotFoundError, ValueError) as exc:
        item_id_mapping_source = "synthetic_item_idx"
        idx_to_item = build_fallback_item_index_mapping(item_latent_table.shape[0])
        print(f"[warn] {exc}")
        print(
            "[warn] Falling back to synthetic item identifiers because the buffer item_map "
            "is unavailable or incompatible. Metrics remain valid; JSONL item_id fields "
            "will use item_idx_<n>."
        )

    item_frequencies = load_split_target_frequencies(
        tokenized_root=tokenized_root,
        split=args.frequency_source_split,
        num_items=item_latent_table.shape[0],
    )
    item_group_ids, grouping_metadata = build_popularity_groups(
        item_frequencies,
        strategy=args.group_strategy,
    )
    popularity_penalty = float(
        args.popularity_penalty
        if args.popularity_penalty is not None
        else evaluation_cfg.get("popularity_penalty", 0.0)
    )
    popularity_penalty_source_split = (
        args.popularity_penalty_source_split
        or evaluation_cfg.get("popularity_penalty_source_split")
        or args.frequency_source_split
    )
    popularity_penalty_vector = None
    if popularity_penalty > 0:
        penalty_frequencies = load_split_target_frequencies(
            tokenized_root=tokenized_root,
            split=popularity_penalty_source_split,
            num_items=item_latent_table.shape[0],
        )
        popularity_penalty_vector = torch.from_numpy(
            build_normalized_popularity_penalty(penalty_frequencies)
        ).to(device=device, dtype=torch.float32)

    if args.save_jsonl is not None and args.save_jsonl.exists():
        args.save_jsonl.unlink()

    overall_tracker = RetrievalMetricTracker(topk_list)
    group_trackers = {group_name: RetrievalMetricTracker(topk_list) for group_name in GROUP_NAMES}
    total_batches = len(dataloader)
    if args.max_eval_batches > 0:
        total_batches = min(total_batches, args.max_eval_batches)

    progress_bar = tqdm(dataloader, total=total_batches, desc="Hybrid Retrieval Eval", unit="batch")
    global_sample_offset = 0

    for step, batch in enumerate(progress_bar, start=1):
        if args.max_eval_batches > 0 and step > args.max_eval_batches:
            break

        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        apply_modality_occlusion_inplace(batch, occluded_modalities)
        sampled_latents = model.sample_latents(
            batch,
            num_inference_steps=args.num_inference_steps,
        ).float()
        if normalize_target_latent:
            sampled_latents = F.normalize(sampled_latents, dim=-1)

        scores = sampled_latents @ item_latent_table.t()

        target_item_ids = batch["target_item_ids"].to(device=device, dtype=torch.long)
        history_item_ids = batch["history_item_ids"].to(device=device, dtype=torch.long)
        history_masks = batch["history_masks"].to(device=device, dtype=torch.bool)

        if args.exclude_history_items:
            scores = exclude_history_items_from_scores(
                scores=scores,
                history_item_ids=history_item_ids,
                history_masks=history_masks,
                target_item_ids=target_item_ids,
            )
        if popularity_penalty_vector is not None:
            scores = scores - float(popularity_penalty) * popularity_penalty_vector.unsqueeze(0)

        target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
        ranks = 1 + (scores > target_scores).sum(dim=1)
        overall_tracker.update(ranks)

        batch_group_ids = torch.from_numpy(item_group_ids[target_item_ids.detach().cpu().numpy()]).to(torch.long)
        for group_id, group_name in enumerate(GROUP_NAMES):
            group_mask = batch_group_ids == group_id
            if group_mask.any():
                group_trackers[group_name].update(ranks[group_mask.to(device=ranks.device)])

        topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=1)
        running_metrics = overall_tracker.compute()
        postfix = {"mean_rank": f"{running_metrics.get('mean_rank', 0.0):.2f}"}
        for k in topk_list:
            postfix[f"ndcg@{k}"] = f"{running_metrics.get(f'ndcg@{k}', 0.0):.4f}"
            postfix[f"hit@{k}"] = f"{running_metrics.get(f'hit@{k}', 0.0):.4f}"
        progress_bar.set_postfix(postfix)

        if args.save_jsonl is not None:
            records: List[dict] = []
            for row_idx in range(target_item_ids.shape[0]):
                row_target_item_idx = int(target_item_ids[row_idx].item())
                row_target_item_id = idx_to_item[row_target_item_idx]
                row_history = []
                for hist_idx, hist_valid in zip(
                    history_item_ids[row_idx].tolist(),
                    history_masks[row_idx].tolist(),
                ):
                    if bool(hist_valid) and int(hist_idx) >= 0:
                        row_history.append(idx_to_item[int(hist_idx)])
                row_retrieved = []
                for rank_pos in range(max_k):
                    item_idx = int(topk_indices[row_idx, rank_pos].item())
                    row_retrieved.append(
                        {
                            "rank": rank_pos + 1,
                            "item_idx": item_idx,
                            "item_id": idx_to_item[item_idx],
                            "score": float(topk_scores[row_idx, rank_pos].item()),
                        }
                    )
                global_index = global_sample_offset + row_idx
                records.append(
                    {
                        "split": args.split,
                        "sample_index": int(global_index),
                        "target_item_idx": row_target_item_idx,
                        "target_item_id": row_target_item_id,
                        "target_rank": int(ranks[row_idx].item()),
                        "history_items": row_history,
                        "retrieved_topk": row_retrieved,
                        "group": GROUP_NAMES[int(batch_group_ids[row_idx].item())],
                    }
                )
            write_jsonl_records(args.save_jsonl, records)

        if args.print_every > 0 and step % args.print_every == 0:
            print(f"[batch {step}/{total_batches}] {json.dumps(running_metrics, ensure_ascii=False)}")

        global_sample_offset += target_item_ids.shape[0]

    overall_metrics = overall_tracker.compute()
    group_metrics = {}
    for group_name in GROUP_NAMES:
        group_metrics[group_name] = {
            "name_zh": GROUP_NAMES_ZH[group_name],
            "sample_count": int(group_trackers[group_name].total),
            **group_trackers[group_name].compute(),
        }

    final_payload = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "tokenized_root": str(tokenized_root),
        "buffer_root": str(buffer_root),
        "target_latent_path": str(target_latent_path),
        "item_id_mapping_source": item_id_mapping_source,
        "evaluated_samples": int(overall_tracker.total),
        "candidate_items": int(item_latent_table.shape[0]),
        "topk": topk_list,
        "exclude_history_items": bool(args.exclude_history_items),
        "num_inference_steps": int(args.num_inference_steps),
        "occluded_modalities": occluded_modalities,
        "use_popularity": use_popularity,
        "popularity_num_buckets": popularity_num_buckets if use_popularity else None,
        "popularity_penalty": float(popularity_penalty),
        "popularity_penalty_source_split": (
            popularity_penalty_source_split if popularity_penalty > 0 else None
        ),
        "overall_metrics": overall_metrics,
        "grouping": {
            "reference": (
                "Grouped by item observation frequency, following the report's idea of "
                "using item occurrence count as frequency."
            ),
            "frequency_source_split": args.frequency_source_split,
            **grouping_metadata,
        },
        "group_metrics": group_metrics,
    }

    plot_path = resolve_plot_path(args)
    if plot_path is not None:
        saved_plot_path = plot_grouped_metrics(
            save_path=plot_path,
            split=args.split,
            topk_list=topk_list,
            overall_metrics=overall_metrics,
            group_metrics=group_metrics,
        )
        final_payload["saved_plot"] = str(saved_plot_path)

    print("")
    print("========== Hybrid Diffusion Final Evaluation ==========")
    print(json.dumps(final_payload, indent=2, ensure_ascii=False))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as fp:
            json.dump(final_payload, fp, indent=2, ensure_ascii=False)
        print(f"saved_json = {args.save_json}")

    if args.save_jsonl is not None:
        print(f"saved_jsonl = {args.save_jsonl}")
    if plot_path is not None:
        print(f"saved_plot = {plot_path}")


if __name__ == "__main__":
    main()
