from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def _parse_topk(topk_arg):
    if topk_arg is None:
        topk_arg = "5,10,20"

    if isinstance(topk_arg, (list, tuple)):
        values = sorted({int(value) for value in topk_arg})
    else:
        topk_text = str(topk_arg).strip()
        if not topk_text:
            return []
        values = sorted({int(item) for item in topk_text.split(",") if item.strip()})

    if any(value <= 0 for value in values):
        raise ValueError(
            "`sample_topk` must only contain positive integers, e.g. `5,10,20`."
        )
    return values


def _compute_scores(pred_embed, item_library, similarity):
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

    raise ValueError(f"Unsupported retrieval similarity: {similarity}")


def _exclude_history_items_from_scores(scores, history_item_ids, history_masks, target_item_ids):
    scores = scores.clone()
    target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
    valid_history = history_masks & history_item_ids.ge(0)
    if valid_history.any():
        row_idx = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        row_idx = row_idx.expand_as(history_item_ids)
        col_idx = history_item_ids.clamp_min(0)
        scores[row_idx[valid_history], col_idx[valid_history]] = -torch.inf
        scores.scatter_(1, target_item_ids.unsqueeze(1), target_scores)
    return scores


def _update_rank_metric_sums(metric_sums, metric_counts, prefix, ranks, topk_list):
    ranks = ranks.detach().cpu().to(torch.float32).reshape(-1)
    if ranks.numel() == 0:
        return

    sample_count = int(ranks.numel())
    metric_sums[f"{prefix}_sample_mean_rank"] += float(ranks.sum().item())
    metric_counts[f"{prefix}_sample_mean_rank"] += sample_count

    for k in topk_list:
        hit = (ranks <= k).float()
        mrr = torch.where(
            ranks <= k,
            1.0 / ranks,
            torch.zeros_like(ranks),
        )
        ndcg = torch.where(
            ranks <= k,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        )

        metric_sums[f"{prefix}_sample_hit_at_{k}"] += float(hit.sum().item())
        metric_counts[f"{prefix}_sample_hit_at_{k}"] += sample_count

        metric_sums[f"{prefix}_sample_mrr_at_{k}"] += float(mrr.sum().item())
        metric_counts[f"{prefix}_sample_mrr_at_{k}"] += sample_count

        metric_sums[f"{prefix}_sample_ndcg_at_{k}"] += float(ndcg.sum().item())
        metric_counts[f"{prefix}_sample_ndcg_at_{k}"] += sample_count


def _warn_once(dataset, attr_name, logger, message):
    if getattr(dataset, attr_name, False):
        return
    logger.info(message)
    setattr(dataset, attr_name, True)


def _build_retrieval_eval_context(args, dataloader, device, logger, model=None):
    topk_list = _parse_topk(getattr(args, "sample_topk", "5,10,20"))
    if not topk_list:
        return None

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not getattr(dataset, "buffer_mode", False):
        if dataset is not None:
            _warn_once(
                dataset,
                "_sample_retrieval_skip_logged",
                logger,
                "Skipping retrieval metrics during sampling because the dataset is not in buffer mode.",
            )
        return None

    buffer_root = getattr(dataset, "buffer_root", None)
    if buffer_root is None:
        _warn_once(
            dataset,
            "_sample_retrieval_skip_logged",
            logger,
            "Skipping retrieval metrics during sampling because `buffer_root` is unavailable.",
        )
        return None

    similarity = getattr(args, "sample_similarity", "cosine")
    if similarity not in {"cosine", "dot", "neg_l2"}:
        raise ValueError(
            "`sample_similarity` must be one of `cosine`, `dot`, or `neg_l2`."
        )

    use_model_item_latents = bool(
        model is not None
        and hasattr(model, "get_item_latent_table")
        and model.get_item_latent_table() is not None
    )
    cache = getattr(dataset, "_sample_retrieval_cache", {})
    cache_key = (str(device), similarity, use_model_item_latents)
    item_library = None if use_model_item_latents else cache.get(cache_key)

    if item_library is None:
        if use_model_item_latents:
            item_library = model.get_item_latent_table(
                normalize=(similarity == "cosine"),
                device=device,
                dtype=torch.float32,
            )
        else:
            item_embeddings = np.load(buffer_root / "item_embeddings.npy", mmap_mode="r")
            item_library = torch.from_numpy(np.asarray(item_embeddings, dtype=np.float32).copy())
            if similarity == "cosine":
                item_library = F.normalize(item_library, dim=-1)
            item_library = item_library.to(device=device)
            cache[cache_key] = item_library
            dataset._sample_retrieval_cache = cache

    _warn_once(
        dataset,
        "_sample_retrieval_enabled_logged",
        logger,
        "Enabled retrieval metrics during sampling with "
        f"topk={topk_list}, similarity={similarity}, "
        f"exclude_history_items={bool(getattr(args, 'sample_exclude_history_items', False))}.",
    )

    return {
        "topk_list": topk_list,
        "similarity": similarity,
        "item_library": item_library,
        "exclude_history_items": bool(
            getattr(args, "sample_exclude_history_items", False)
        ),
    }


@torch.no_grad()
def log_sample_res(
    rdt,
    args,
    accelerator,
    weight_dtype,
    dataset_id2name,
    dataloader,
    logger,
):
    logger.info(
        f"Running sampling for {args.num_sample_batches} batches..."
    )

    rdt.eval()
    model = accelerator.unwrap_model(rdt)

    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    retrieval_eval = _build_retrieval_eval_context(
        args=args,
        dataloader=dataloader,
        device=accelerator.device,
        logger=logger,
        model=model,
    )

    for step, batch in enumerate(dataloader):
        if step >= args.num_sample_batches:
            break

        data_indices = batch["data_indices"]
        history_id_embeds = batch["history_id_embeds"].to(
            device=accelerator.device,
            dtype=weight_dtype,
        )
        history_pixel_values = batch["history_pixel_values"].to(
            device=accelerator.device,
            dtype=weight_dtype,
        )

        pred_actions = model.sample(
            history_id_embeds=history_id_embeds,
            history_pixel_values=history_pixel_values,
            text=batch["text"],
        )

        if (
            "target_item_ids" in batch
            and hasattr(model, "lookup_item_latents")
            and getattr(model, "has_trainable_item_latents", lambda: False)()
        ):
            target_item_ids_for_loss = batch["target_item_ids"].to(
                device=accelerator.device,
                dtype=torch.long,
            )
            target_embed = model.lookup_item_latents(
                target_item_ids_for_loss,
                dtype=weight_dtype,
            ).unsqueeze(1)
        else:
            target_embed = batch["target_embed"].to(
                device=accelerator.device,
                dtype=weight_dtype,
            )

        loss = F.mse_loss(pred_actions, target_embed, reduction="none").float()
        mse_loss_per_entry = loss.reshape(loss.shape[0], -1).mean(dim=1)
        l2_loss_per_entry = loss.sqrt().reshape(loss.shape[0], -1).mean(dim=1)

        dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics(
            (
                data_indices.to(device=pred_actions.device, dtype=torch.long),
                mse_loss_per_entry,
                l2_loss_per_entry,
            ),
        )

        metric_sums["overall_avg_sample_mse"] += float(mse_losses.sum().item())
        metric_counts["overall_avg_sample_mse"] += int(mse_losses.numel())
        metric_sums["overall_avg_sample_l2err"] += float(l2_losses.sum().item())
        metric_counts["overall_avg_sample_l2err"] += int(l2_losses.numel())

        dataset_indices = dataset_indices.tolist()
        for loss_suffix, losses in (
            ("_sample_mse", mse_losses),
            ("_sample_l2err", l2_losses),
        ):
            for dataset_idx, loss_tensor in zip(dataset_indices, losses):
                loss_name = dataset_id2name[dataset_idx] + loss_suffix
                metric_sums[loss_name] += float(loss_tensor.item())
                metric_counts[loss_name] += 1

        if retrieval_eval is None:
            continue

        if any(
            key not in batch for key in ("target_item_ids", "history_item_ids", "history_masks")
        ):
            retrieval_eval = None
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None:
                _warn_once(
                    dataset,
                    "_sample_retrieval_skip_missing_fields_logged",
                    logger,
                    "Skipping retrieval metrics during sampling because the batch "
                    "does not contain `target_item_ids`, `history_item_ids`, and `history_masks`.",
                )
            continue

        target_item_ids = batch["target_item_ids"].to(
            device=accelerator.device,
            dtype=torch.long,
        )
        if (target_item_ids < 0).any():
            retrieval_eval = None
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None:
                _warn_once(
                    dataset,
                    "_sample_retrieval_skip_invalid_target_logged",
                    logger,
                    "Skipping retrieval metrics during sampling because "
                    "`target_item_ids` contains invalid entries.",
                )
            continue

        history_item_ids = batch["history_item_ids"].to(
            device=accelerator.device,
            dtype=torch.long,
        )
        history_masks = batch["history_masks"].to(
            device=accelerator.device,
            dtype=torch.bool,
        )

        scores = _compute_scores(
            pred_embed=pred_actions,
            item_library=retrieval_eval["item_library"],
            similarity=retrieval_eval["similarity"],
        )
        if retrieval_eval["exclude_history_items"]:
            scores = _exclude_history_items_from_scores(
                scores=scores,
                history_item_ids=history_item_ids,
                history_masks=history_masks,
                target_item_ids=target_item_ids,
            )

        target_scores = scores.gather(1, target_item_ids.unsqueeze(1))
        ranks = 1 + (scores > target_scores).sum(dim=1)
        gathered_ranks = accelerator.gather_for_metrics(ranks.to(torch.long))
        _update_rank_metric_sums(
            metric_sums=metric_sums,
            metric_counts=metric_counts,
            prefix="overall_avg",
            ranks=gathered_ranks,
            topk_list=retrieval_eval["topk_list"],
        )

    for name, value in list(metric_sums.items()):
        count = metric_counts.get(name, 0)
        if count <= 0:
            del metric_sums[name]
            continue
        metric_sums[name] = round(value / count, 4)

    rdt.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(metric_sums)
