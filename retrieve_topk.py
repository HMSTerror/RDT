#!/usr/bin/env python
# coding=utf-8

import argparse
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from models.multimodal_encoder.clip_encoder import CLIPVisionTower
from models.multimodal_encoder.condition_encoder import ConditionEncoder
from models.multimodal_encoder.dummy_encoder import DummyVisionTower
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k Amazon items from predicted [B,1,128] embeddings and evaluate ranking metrics."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/recsys_amazon.yaml",
        help="Path to the RecSys-DiT YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file or directory to load.",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        required=True,
        help="Path or HF id of the T5 text encoder.",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        required=True,
        help="Path or HF id of the SigLIP/CLIP vision encoder.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=0,
        help="Optional max number of evaluation batches. 0 means full dataset.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="5,10,20",
        help="Comma-separated top-k list, e.g. '5,10,20,50'.",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        default="cosine",
        choices=["cosine", "dot", "neg_l2"],
        help="Similarity used to retrieve items from the item embedding library.",
    )
    parser.add_argument(
        "--exclude_history_items",
        action="store_true",
        help="Exclude history items from the candidate pool. Disabled by default because the target can repeat.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run retrieval on. Defaults to cuda if available, otherwise cpu.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--save_jsonl",
        type=str,
        default=None,
        help="Optional JSONL path for saving per-sample retrieval results.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print running metrics every N evaluation batches.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_weight_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)

    buffer_root = (
        config.get("dataset", {}).get("buffer_root")
        or config.get("dataset", {}).get("preprocessed_buffer_root")
    )
    if buffer_root:
        buffer_root = Path(buffer_root)
        if not buffer_root.is_absolute():
            buffer_root = Path.cwd() / buffer_root
        stats_path = buffer_root / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as fp:
                stats = json.load(fp)
            config["dataset"]["history_len"] = int(stats["history_len"])
            config["common"]["img_history_size"] = int(stats["history_len"])

    config["common"]["state_dim"] = 128
    config["common"]["action_chunk_size"] = 1
    config["model"]["state_token_dim"] = 128
    config["model"]["rdt"]["hidden_size"] = 1024
    return config


def build_vision_encoder(vision_tower_name: str, config: dict):
    if ("clip" in vision_tower_name.lower()) and ("siglip" not in vision_tower_name.lower()):
        clip_args = SimpleNamespace(mm_vision_select_layer=-2, mm_vision_select_feature="patch")
        return CLIPVisionTower(vision_tower=vision_tower_name, args=clip_args)
    if vision_tower_name.lower() == "dummy":
        return DummyVisionTower(hidden_size=config["model"]["img_token_dim"])
    return SiglipVisionTower(vision_tower=vision_tower_name, args=None)


def build_model_and_dataset(args):
    config = load_config(args.config_path)
    device = resolve_device(args.device)
    weight_dtype = resolve_weight_dtype(args.mixed_precision, device)

    text_embedder = T5Embedder(
        from_pretrained=args.pretrained_text_encoder_name_or_path,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        torch_dtype=weight_dtype,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = build_vision_encoder(args.pretrained_vision_encoder_name_or_path, config)
    image_processor = vision_encoder.image_processor

    history_len = max(
        1,
        int(config["dataset"].get("history_len", config["common"]["img_history_size"])),
    )
    condition_encoder = ConditionEncoder(
        history_len=history_len,
        hidden_dim=1024,
        vision_encoder=vision_encoder,
        text_tokenizer=tokenizer,
        text_encoder=text_encoder,
        max_text_length=64,
        device=device,
        torch_dtype=weight_dtype,
    )

    runner = RDTRunner(
        action_dim=config["common"]["state_dim"],
        pred_horizon=config["common"]["action_chunk_size"],
        config=config["model"],
        dtype=weight_dtype,
    ).set_condition_encoder(condition_encoder)

    load_checkpoint(runner, args.checkpoint)
    runner.to(device)
    runner.eval()

    dataset_config = copy.deepcopy(config["dataset"])
    dataset_config["text_branch_mask_prob"] = 0.0
    dataset_config["image_branch_mask_prob"] = 0.0
    dataset = VLAConsumerDataset(
        config=dataset_config,
        image_processor=image_processor,
        img_history_size=config["common"]["img_history_size"],
        cond_mask_prob=0.0,
        image_aug=False,
        auto_adjust_image_brightness=False,
    )
    if not dataset.buffer_mode:
        raise RuntimeError(
            "retrieve_topk.py currently requires buffer mode. Set dataset.buffer_root "
            "to a lightweight preprocessed buffer directory."
        )

    collator = DataCollatorForVLAConsumerDataset()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.dataloader_num_workers > 0),
    )
    return config, device, weight_dtype, runner, dataset, dataloader


def resolve_checkpoint_file(checkpoint_path: str) -> Path:
    path = Path(checkpoint_path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    candidates = [
        path / "pytorch_model.bin",
        path / "model.safetensors",
        path / "diffusion_pytorch_model.bin",
        path / "diffusion_pytorch_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find a supported checkpoint file inside directory: {checkpoint_path}"
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    checkpoint_file = resolve_checkpoint_file(checkpoint_path)
    suffix = checkpoint_file.suffix.lower()

    if suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(checkpoint_file))
    else:
        payload = torch.load(checkpoint_file, map_location="cpu")
        if isinstance(payload, dict) and "module" in payload:
            state_dict = payload["module"]
        else:
            state_dict = payload

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[warn] Missing keys when loading checkpoint: {len(missing_keys)}")
        preview = missing_keys[:10]
        print(f"[warn] Missing preview: {preview}")
    if unexpected_keys:
        print(f"[warn] Unexpected keys when loading checkpoint: {len(unexpected_keys)}")
        preview = unexpected_keys[:10]
        print(f"[warn] Unexpected preview: {preview}")


def parse_topk(topk_arg: str) -> List[int]:
    values = sorted({int(item) for item in topk_arg.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise ValueError("--topk must contain positive integers, e.g. '5,10,20'.")
    return values


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

    def update(self, ranks: torch.Tensor):
        ranks = ranks.detach().cpu().to(torch.long)
        batch_size = int(ranks.shape[0])
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
            self.sums[f"recall@{k}"] += float(hit.sum().item())  # one relevant item per query
            self.sums[f"mrr@{k}"] += float(mrr.sum().item())
            self.sums[f"ndcg@{k}"] += float(ndcg.sum().item())

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {}
        result = {}
        for name, value in self.sums.items():
            result[name] = value / self.total
        return result


def prepare_candidate_library(dataset, device: torch.device, similarity: str):
    item_embeddings = np.load(dataset.buffer_root / "item_embeddings.npy", mmap_mode="r")
    item_tensor = torch.from_numpy(np.asarray(item_embeddings, dtype=np.float32))
    if similarity == "cosine":
        item_tensor = F.normalize(item_tensor, dim=-1)
    return item_tensor.to(device)


def build_idx_to_item_and_meta(dataset):
    with open(dataset.buffer_root / "item_map.json", "r", encoding="utf-8") as fp:
        item_map = json.load(fp)
    idx_to_item = {int(idx): item_id for item_id, idx in item_map.items()}
    with open(dataset.buffer_root / "item_meta.json", "r", encoding="utf-8") as fp:
        item_meta = json.load(fp)
    return idx_to_item, item_meta


def compute_scores(pred_embed, item_library, similarity: str):
    pred = pred_embed[:, 0, :].float()
    candidates = item_library.float()
    if similarity == "cosine":
        pred = F.normalize(pred, dim=-1)
        return pred @ candidates.t()
    if similarity == "dot":
        return pred @ candidates.t()
    if similarity == "neg_l2":
        pred_sq = (pred ** 2).sum(dim=-1, keepdim=True)
        item_sq = (candidates ** 2).sum(dim=-1).unsqueeze(0)
        return -(pred_sq + item_sq - 2 * (pred @ candidates.t()))
    raise ValueError(f"Unsupported similarity: {similarity}")


def exclude_history_items_from_scores(scores, history_item_ids, history_masks, target_item_ids):
    scores = scores.clone()
    target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
    valid_history = history_masks & history_item_ids.ge(0)
    if valid_history.any():
        row_idx = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(history_item_ids)
        col_idx = history_item_ids.clamp_min(0)
        scores[row_idx[valid_history], col_idx[valid_history]] = -torch.inf
        scores.scatter_(1, target_item_ids.unsqueeze(1), target_scores)
    return scores


def write_jsonl_records(path: Path, records: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def main():
    args = parse_args()
    set_seed(args.seed)

    topk_list = parse_topk(args.topk)
    max_k = max(topk_list)

    config, device, weight_dtype, runner, dataset, dataloader = build_model_and_dataset(args)
    item_library = prepare_candidate_library(dataset, device=device, similarity=args.similarity)
    idx_to_item, item_meta = build_idx_to_item_and_meta(dataset)

    if args.save_jsonl:
        save_path = Path(args.save_jsonl)
        if save_path.exists():
            save_path.unlink()
    else:
        save_path = None

    tracker = RetrievalMetricTracker(topk_list)
    total_batches = len(dataloader)
    if args.max_eval_batches > 0:
        total_batches = min(total_batches, args.max_eval_batches)

    progress_bar = tqdm(dataloader, total=total_batches, desc="Retrieval Eval", unit="batch")
    global_sample_offset = 0

    for step, batch in enumerate(progress_bar, start=1):
        if args.max_eval_batches > 0 and step > args.max_eval_batches:
            break

        history_id_embeds = batch["history_id_embeds"].to(device=device, dtype=weight_dtype)
        history_pixel_values = batch["history_pixel_values"].to(device=device, dtype=weight_dtype)
        target_item_ids = batch["target_item_ids"].to(device=device, dtype=torch.long)
        history_item_ids = batch["history_item_ids"].to(device=device, dtype=torch.long)
        history_masks = batch["history_masks"].to(device=device, dtype=torch.bool)

        if (target_item_ids < 0).any():
            raise RuntimeError(
                "retrieve_topk.py requires valid target_item_ids. Please evaluate with buffer mode."
            )

        pred_embed = runner.sample(
            history_id_embeds=history_id_embeds,
            history_pixel_values=history_pixel_values,
            text=batch["text"],
        )
        scores = compute_scores(pred_embed, item_library=item_library, similarity=args.similarity)

        if args.exclude_history_items:
            scores = exclude_history_items_from_scores(
                scores=scores,
                history_item_ids=history_item_ids,
                history_masks=history_masks,
                target_item_ids=target_item_ids,
            )

        target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
        ranks = 1 + (scores > target_scores).sum(dim=1)
        tracker.update(ranks)

        topk_scores, topk_indices = torch.topk(scores, k=max_k, dim=1)

        if save_path is not None:
            records = []
            for row_idx in range(target_item_ids.shape[0]):
                target_item_id = int(target_item_ids[row_idx].item())
                target_asin = idx_to_item[target_item_id]
                predictions = []
                for rank_pos in range(max_k):
                    item_id = int(topk_indices[row_idx, rank_pos].item())
                    asin = idx_to_item[item_id]
                    meta = item_meta.get(asin, {})
                    predictions.append(
                        {
                            "rank": rank_pos + 1,
                            "item_id": item_id,
                            "asin": asin,
                            "title": meta.get("title", ""),
                            "categories": meta.get("categories", ""),
                            "score": float(topk_scores[row_idx, rank_pos].item()),
                        }
                    )

                target_meta = item_meta.get(target_asin, {})
                records.append(
                    {
                        "sample_index": global_sample_offset + row_idx,
                        "target_item_id": target_item_id,
                        "target_asin": target_asin,
                        "target_title": target_meta.get("title", ""),
                        "target_categories": target_meta.get("categories", ""),
                        "rank": int(ranks[row_idx].item()),
                        "topk": predictions,
                    }
                )
            write_jsonl_records(save_path, records)

        running_metrics = tracker.compute()
        postfix = {"mean_rank": f"{running_metrics['mean_rank']:.2f}"}
        for k in topk_list:
            postfix[f"ndcg@{k}"] = f"{running_metrics[f'ndcg@{k}']:.4f}"
            postfix[f"hit@{k}"] = f"{running_metrics[f'hit@{k}']:.4f}"
        progress_bar.set_postfix(postfix)

        if args.print_every > 0 and (step % args.print_every == 0):
            print(f"[batch {step}/{total_batches}] {json.dumps(running_metrics, ensure_ascii=False)}")

        global_sample_offset += target_item_ids.shape[0]

    final_metrics = tracker.compute()
    print("")
    print("========== Final Retrieval Metrics ==========")
    print(json.dumps(final_metrics, indent=2, ensure_ascii=False))
    print("")
    print(f"evaluated_samples = {tracker.total}")
    print(f"candidate_items   = {item_library.shape[0]}")
    print(f"similarity        = {args.similarity}")
    print(f"topk              = {topk_list}")
    if save_path is not None:
        print(f"saved_predictions = {save_path}")


if __name__ == "__main__":
    main()
