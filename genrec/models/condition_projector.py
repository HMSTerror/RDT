from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ConditionProjectorOutput:
    tokens: torch.Tensor | None
    attention_mask: torch.Tensor | None
    history_token_count: int = 0
    pooled_token_count: int = 0
    target_token_count: int = 0


class ConditionBranchProjector(nn.Module):
    """
    Project one dense condition branch into token-level conditioning features.

    The branch can expose three token groups:
    - history tokens: one token per interacted item
    - pooled token: one summary token from the valid history
    - target token: optional target-side token, kept disabled by default to avoid
      leaking the ground-truth item into next-item prediction
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        max_history_len: int,
        use_history: bool = True,
        use_pooled: bool = True,
        use_target: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.max_history_len = int(max_history_len)
        self.use_history = bool(use_history)
        self.use_pooled = bool(use_pooled)
        self.use_target = bool(use_target)

        self.history_proj = nn.Linear(self.input_dim, self.hidden_size) if self.use_history else None
        self.pooled_proj = nn.Linear(self.input_dim, self.hidden_size) if self.use_pooled else None
        self.target_proj = nn.Linear(self.input_dim, self.hidden_size) if self.use_target else None

        self.history_pos_embed = nn.Embedding(self.max_history_len, self.hidden_size)
        self.token_type_embed = nn.Embedding(3, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.hidden_size)

    def forward(
        self,
        *,
        history_embeds: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        pooled_embed: torch.Tensor | None = None,
        target_embed: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> ConditionProjectorOutput:
        tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []

        batch_size = None
        device = None
        dtype = None

        for candidate in (history_embeds, pooled_embed, target_embed):
            if candidate is not None:
                batch_size = candidate.shape[0]
                device = candidate.device
                dtype = candidate.dtype
                break

        if batch_size is None:
            return ConditionProjectorOutput(tokens=None, attention_mask=None)

        history_token_count = 0
        pooled_token_count = 0
        target_token_count = 0

        if self.use_history and history_embeds is not None:
            if history_embeds.ndim != 3:
                raise ValueError(
                    f"`history_embeds` must have shape [B, H, D], got {tuple(history_embeds.shape)}."
                )
            if history_embeds.shape[1] > self.max_history_len:
                raise ValueError(
                    f"history length {history_embeds.shape[1]} exceeds max_history_len={self.max_history_len}."
                )

            hist_tokens = self.history_proj(history_embeds)
            positions = torch.arange(
                history_embeds.shape[1],
                device=history_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)
            hist_tokens = hist_tokens + self.history_pos_embed(positions)
            hist_tokens = hist_tokens + self.token_type_embed.weight[0].view(1, 1, -1)
            tokens.append(hist_tokens)
            history_token_count = int(history_embeds.shape[1])

            if history_mask is None:
                history_mask = torch.ones(
                    history_embeds.shape[:2],
                    device=history_embeds.device,
                    dtype=torch.bool,
                )
            else:
                history_mask = history_mask.to(device=history_embeds.device, dtype=torch.bool)
            masks.append(history_mask)

        valid_history_mask = None
        if history_mask is not None:
            valid_history_mask = history_mask.any(dim=1, keepdim=True)

        if self.use_pooled and pooled_embed is not None:
            if pooled_embed.ndim != 2:
                raise ValueError(
                    f"`pooled_embed` must have shape [B, D], got {tuple(pooled_embed.shape)}."
                )
            pooled_tokens = self.pooled_proj(pooled_embed).unsqueeze(1)
            pooled_tokens = pooled_tokens + self.token_type_embed.weight[1].view(1, 1, -1)
            tokens.append(pooled_tokens)
            pooled_token_count = 1

            pooled_mask = valid_history_mask
            if pooled_mask is None:
                pooled_mask = torch.ones(
                    (batch_size, 1),
                    device=device,
                    dtype=torch.bool,
                )
            masks.append(pooled_mask)

        if self.use_target and target_embed is not None:
            if target_embed.ndim != 2:
                raise ValueError(
                    f"`target_embed` must have shape [B, D], got {tuple(target_embed.shape)}."
                )
            target_tokens = self.target_proj(target_embed).unsqueeze(1)
            target_tokens = target_tokens + self.token_type_embed.weight[2].view(1, 1, -1)
            tokens.append(target_tokens)
            target_token_count = 1

            if target_mask is None:
                target_mask = torch.ones(
                    (batch_size, 1),
                    device=device,
                    dtype=torch.bool,
                )
            else:
                target_mask = target_mask.to(device=device, dtype=torch.bool).view(batch_size, 1)
            masks.append(target_mask)

        if not tokens:
            return ConditionProjectorOutput(tokens=None, attention_mask=None)

        projected = torch.cat(tokens, dim=1).to(dtype=dtype)
        projected = self.dropout(self.norm(projected))
        attention_mask = torch.cat(masks, dim=1)
        return ConditionProjectorOutput(
            tokens=projected,
            attention_mask=attention_mask,
            history_token_count=history_token_count,
            pooled_token_count=pooled_token_count,
            target_token_count=target_token_count,
        )
