from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from timm.models.vision_transformer import RmsNorm

from genrec.contracts import SemanticIdBatch
from models.rdt.blocks import TimestepEmbedder

from .condition_projector import ConditionBranchProjector, ConditionProjectorOutput
from .genrec_dit import GenRecDiTBlock


@dataclass
class HybridDiffusionOutput:
    prediction: torch.Tensor
    denoised_latents: torch.Tensor
    target_positions: torch.Tensor
    hidden_states: torch.Tensor
    text_aux_query: torch.Tensor | None = None
    image_aux_query: torch.Tensor | None = None
    text_history_aux_query: torch.Tensor | None = None
    image_history_aux_query: torch.Tensor | None = None


@dataclass
class BranchLayerSchedule:
    text: tuple[bool, ...]
    image: tuple[bool, ...]
    cf: tuple[bool, ...]
    popularity: tuple[bool, ...]


class GenRecHybridDiffusionRunner(nn.Module):
    """
    Hybrid GenRec + Diffusion runner:
    - history/context is still semantic-ID tokenized
    - target is a continuous latent denoised with diffusion
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        vocab_sizes: list[int],
        max_seq_len: int,
        max_history_len: int,
        latent_dim: int = 128,
        prediction_type: str = "epsilon",
        num_train_timesteps: int = 1000,
        beta_schedule: str = "squaredcos_cap_v2",
        hidden_size: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        num_special_tokens: int = 5,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        text_cond_dim: int | None = None,
        image_cond_dim: int | None = None,
        cf_cond_dim: int | None = None,
        popularity_cond_dim: int | None = None,
        popularity_num_buckets: int | None = None,
        item_popularity_bucket_ids: torch.Tensor | None = None,
        use_text_history: bool = True,
        use_text_pooled: bool = True,
        use_text_target: bool = False,
        use_image_history: bool = True,
        use_image_pooled: bool = True,
        use_image_target: bool = False,
        use_cf_history: bool = True,
        use_cf_pooled: bool = True,
        use_cf_target: bool = False,
        use_popularity_history: bool = True,
        use_popularity_pooled: bool = True,
        text_injection_mode: str = "all",
        image_injection_mode: str = "all",
        cf_injection_mode: str = "all",
        popularity_injection_mode: str = "all",
        text_branch_dropout: float = 0.0,
        image_branch_dropout: float = 0.0,
        cf_branch_dropout: float = 0.0,
        popularity_branch_dropout: float = 0.0,
        text_history_token_keep_rate: float = 1.0,
        image_history_token_keep_rate: float = 1.0,
        cf_history_token_keep_rate: float = 1.0,
        popularity_history_token_keep_rate: float = 1.0,
        keep_pooled_condition_tokens: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.vocab_sizes = [int(value) for value in vocab_sizes]
        self.code_len = len(self.vocab_sizes)
        self.max_seq_len = int(max_seq_len)
        self.max_history_len = int(max_history_len)
        self.hidden_size = int(hidden_size)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.num_special_tokens = int(num_special_tokens)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)
        self.latent_dim = int(latent_dim)
        self.prediction_type = str(prediction_type)
        if self.prediction_type not in {"epsilon", "sample"}:
            raise ValueError(f"Unsupported prediction_type={self.prediction_type}. Use `epsilon` or `sample`.")
        if self.latent_dim % self.code_len != 0:
            raise ValueError(
                f"latent_dim={self.latent_dim} must be divisible by code_len={self.code_len}."
            )
        self.latent_slot_dim = self.latent_dim // self.code_len
        self.layer_schedule = BranchLayerSchedule(
            text=self._resolve_injection_schedule(text_injection_mode),
            image=self._resolve_injection_schedule(image_injection_mode),
            cf=self._resolve_injection_schedule(cf_injection_mode),
            popularity=self._resolve_injection_schedule(popularity_injection_mode),
        )
        self.branch_dropout_probs = {
            "text": self._validate_probability(text_branch_dropout, "text_branch_dropout"),
            "image": self._validate_probability(image_branch_dropout, "image_branch_dropout"),
            "cf": self._validate_probability(cf_branch_dropout, "cf_branch_dropout"),
            "popularity": self._validate_probability(popularity_branch_dropout, "popularity_branch_dropout"),
        }
        self.history_token_keep_rates = {
            "text": self._validate_probability(text_history_token_keep_rate, "text_history_token_keep_rate"),
            "image": self._validate_probability(image_history_token_keep_rate, "image_history_token_keep_rate"),
            "cf": self._validate_probability(cf_history_token_keep_rate, "cf_history_token_keep_rate"),
            "popularity": self._validate_probability(
                popularity_history_token_keep_rate,
                "popularity_history_token_keep_rate",
            ),
        }
        self.keep_pooled_condition_tokens = bool(keep_pooled_condition_tokens)

        self.token_embed = nn.Embedding(
            self.vocab_size,
            self.hidden_size,
            padding_idx=self.pad_token_id,
        )
        self.position_embed = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.slot_embed = nn.Embedding(self.max_history_len + 2, self.hidden_size)
        self.codebook_embed = nn.Embedding(self.code_len + 2, self.hidden_size)
        self.embed_dropout = nn.Dropout(dropout)
        self.input_norm = RmsNorm(self.hidden_size, eps=1e-6)

        self.target_latent_in = nn.Linear(self.latent_slot_dim, self.hidden_size)
        self.target_latent_out = nn.Linear(self.hidden_size, self.latent_slot_dim)
        self.target_slot_embed = nn.Embedding(self.code_len, self.hidden_size)
        self.timestep_embedder = TimestepEmbedder(self.hidden_size, dtype=torch.float32)
        self.timestep_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.blocks = nn.ModuleList(
            [GenRecDiTBlock(self.hidden_size, self.num_heads, dropout=dropout) for _ in range(self.depth)]
        )
        self.final_norm = RmsNorm(self.hidden_size, eps=1e-6)
        self.text_aux_head = None
        self.image_aux_head = None

        self.text_condition_projector = None
        if text_cond_dim is not None and any(self.layer_schedule.text):
            self.text_condition_projector = ConditionBranchProjector(
                input_dim=int(text_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_text_history,
                use_pooled=use_text_pooled,
                use_target=use_text_target,
                dropout=dropout,
            )
            self.text_aux_head = nn.Linear(self.hidden_size, self.latent_dim)

        self.image_condition_projector = None
        if image_cond_dim is not None and any(self.layer_schedule.image):
            self.image_condition_projector = ConditionBranchProjector(
                input_dim=int(image_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_image_history,
                use_pooled=use_image_pooled,
                use_target=use_image_target,
                dropout=dropout,
            )
            self.image_aux_head = nn.Linear(self.hidden_size, self.latent_dim)

        self.cf_condition_projector = None
        if cf_cond_dim is not None and any(self.layer_schedule.cf):
            self.cf_condition_projector = ConditionBranchProjector(
                input_dim=int(cf_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_cf_history,
                use_pooled=use_cf_pooled,
                use_target=use_cf_target,
                dropout=dropout,
            )

        self.popularity_condition_projector = None
        self.popularity_embedding = None
        self.popularity_padding_bucket_id: int | None = None
        if (
            popularity_cond_dim is not None
            and popularity_num_buckets is not None
            and any(self.layer_schedule.popularity)
        ):
            if item_popularity_bucket_ids is None:
                raise ValueError(
                    "item_popularity_bucket_ids must be provided when enabling popularity conditioning."
                )
            self.popularity_padding_bucket_id = int(popularity_num_buckets)
            self.popularity_embedding = nn.Embedding(
                int(popularity_num_buckets) + 1,
                int(popularity_cond_dim),
                padding_idx=self.popularity_padding_bucket_id,
            )
            self.popularity_condition_projector = ConditionBranchProjector(
                input_dim=int(popularity_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_popularity_history,
                use_pooled=use_popularity_pooled,
                use_target=False,
                dropout=dropout,
            )
            self.register_buffer(
                "item_popularity_bucket_ids",
                item_popularity_bucket_ids.to(dtype=torch.long).clone(),
                persistent=True,
            )
        else:
            self.register_buffer(
                "item_popularity_bucket_ids",
                torch.empty(0, dtype=torch.long),
                persistent=True,
            )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=int(num_train_timesteps),
            beta_schedule=beta_schedule,
            prediction_type=self.prediction_type,
            clip_sample=False,
        )

        self._init_weights()

    def _validate_probability(self, value: float, name: str) -> float:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        return value

    def _resolve_injection_schedule(self, mode: str) -> tuple[bool, ...]:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode == "all":
            return tuple(True for _ in range(self.depth))
        if normalized_mode == "none":
            return tuple(False for _ in range(self.depth))
        if normalized_mode == "front_half":
            cutoff = max(1, self.depth // 2)
            return tuple(layer_idx < cutoff for layer_idx in range(self.depth))
        if normalized_mode == "back_half":
            cutoff = self.depth // 2
            return tuple(layer_idx >= cutoff for layer_idx in range(self.depth))
        if normalized_mode == "every_other":
            return tuple(layer_idx % 2 == 0 for layer_idx in range(self.depth))
        if normalized_mode == "odd_layers":
            return tuple(layer_idx % 2 == 1 for layer_idx in range(self.depth))
        if normalized_mode == "front_sparse":
            cutoff = max(1, self.depth // 2)
            return tuple((layer_idx < cutoff) and (layer_idx % 2 == 0) for layer_idx in range(self.depth))
        if normalized_mode == "back_sparse":
            cutoff = self.depth // 2
            return tuple((layer_idx >= cutoff) and (layer_idx % 2 == 0) for layer_idx in range(self.depth))
        if normalized_mode.startswith("custom:"):
            raw_values = normalized_mode.split(":", 1)[1].strip()
            if not raw_values:
                raise ValueError("custom injection mode requires explicit layer indices, e.g. custom:0,2,4")
            active_layers = set()
            for raw_value in raw_values.split(","):
                value = raw_value.strip()
                if not value:
                    continue
                layer_idx = int(value)
                if not 0 <= layer_idx < self.depth:
                    raise ValueError(
                        f"Custom injection layer index {layer_idx} is out of range for depth={self.depth}."
                    )
                active_layers.add(layer_idx)
            return tuple(layer_idx in active_layers for layer_idx in range(self.depth))
        raise ValueError(
            "Unsupported injection mode "
            f"{mode!r}. Expected one of: all, none, front_half, back_half, "
            "every_other, odd_layers, front_sparse, back_sparse, custom:<idxs>."
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.pad_token_id >= 0:
            with torch.no_grad():
                self.token_embed.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)
        nn.init.normal_(self.codebook_embed.weight, std=0.02)
        nn.init.normal_(self.target_slot_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.target_latent_in.weight)
        nn.init.zeros_(self.target_latent_in.bias)
        nn.init.xavier_uniform_(self.target_latent_out.weight)
        nn.init.zeros_(self.target_latent_out.bias)
        nn.init.xavier_uniform_(self.timestep_proj.weight)
        nn.init.zeros_(self.timestep_proj.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module in {
                    self.target_latent_in,
                    self.target_latent_out,
                    self.timestep_proj,
                }:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_batch_metadata(
        cls,
        *,
        manifest: dict,
        max_history_len: int,
        latent_dim: int,
        prediction_type: str,
        num_train_timesteps: int,
        beta_schedule: str,
        hidden_size: int,
        depth: int,
        num_heads: int,
        text_cond_dim: int | None = None,
        image_cond_dim: int | None = None,
        cf_cond_dim: int | None = None,
        popularity_cond_dim: int | None = None,
        popularity_num_buckets: int | None = None,
        item_popularity_bucket_ids: torch.Tensor | None = None,
        use_text_history: bool = True,
        use_text_pooled: bool = True,
        use_text_target: bool = False,
        use_image_history: bool = True,
        use_image_pooled: bool = True,
        use_image_target: bool = False,
        use_cf_history: bool = True,
        use_cf_pooled: bool = True,
        use_cf_target: bool = False,
        use_popularity_history: bool = True,
        use_popularity_pooled: bool = True,
        text_injection_mode: str = "all",
        image_injection_mode: str = "all",
        cf_injection_mode: str = "all",
        popularity_injection_mode: str = "all",
        text_branch_dropout: float = 0.0,
        image_branch_dropout: float = 0.0,
        cf_branch_dropout: float = 0.0,
        popularity_branch_dropout: float = 0.0,
        text_history_token_keep_rate: float = 1.0,
        image_history_token_keep_rate: float = 1.0,
        cf_history_token_keep_rate: float = 1.0,
        popularity_history_token_keep_rate: float = 1.0,
        keep_pooled_condition_tokens: bool = True,
        dropout: float = 0.0,
    ) -> "GenRecHybridDiffusionRunner":
        special_tokens = manifest.get("special_tokens", {})
        max_seq_len = max(
            int(split_info["max_seq_len"])
            for split_info in manifest.get("splits", {}).values()
        )
        return cls(
            vocab_size=int(manifest["vocab_size"]),
            vocab_sizes=[int(value) for value in manifest["vocab_sizes"]],
            max_seq_len=max_seq_len,
            max_history_len=max_history_len,
            latent_dim=latent_dim,
            prediction_type=prediction_type,
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_special_tokens=len(special_tokens) or 5,
            pad_token_id=int(special_tokens.get("pad_token_id", 0)),
            mask_token_id=int(special_tokens.get("mask_token_id", 1)),
            text_cond_dim=text_cond_dim,
            image_cond_dim=image_cond_dim,
            cf_cond_dim=cf_cond_dim,
            popularity_cond_dim=popularity_cond_dim,
            popularity_num_buckets=popularity_num_buckets,
            item_popularity_bucket_ids=item_popularity_bucket_ids,
            use_text_history=use_text_history,
            use_text_pooled=use_text_pooled,
            use_text_target=use_text_target,
            use_image_history=use_image_history,
            use_image_pooled=use_image_pooled,
            use_image_target=use_image_target,
            use_cf_history=use_cf_history,
            use_cf_pooled=use_cf_pooled,
            use_cf_target=use_cf_target,
            use_popularity_history=use_popularity_history,
            use_popularity_pooled=use_popularity_pooled,
            text_injection_mode=text_injection_mode,
            image_injection_mode=image_injection_mode,
            cf_injection_mode=cf_injection_mode,
            popularity_injection_mode=popularity_injection_mode,
            text_branch_dropout=text_branch_dropout,
            image_branch_dropout=image_branch_dropout,
            cf_branch_dropout=cf_branch_dropout,
            popularity_branch_dropout=popularity_branch_dropout,
            text_history_token_keep_rate=text_history_token_keep_rate,
            image_history_token_keep_rate=image_history_token_keep_rate,
            cf_history_token_keep_rate=cf_history_token_keep_rate,
            popularity_history_token_keep_rate=popularity_history_token_keep_rate,
            keep_pooled_condition_tokens=keep_pooled_condition_tokens,
            dropout=dropout,
        )

    def _build_popularity_branch_inputs(
        self,
        *,
        history_item_ids: torch.Tensor | None,
        history_masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (
            self.popularity_condition_projector is None
            or self.popularity_embedding is None
            or self.item_popularity_bucket_ids.numel() == 0
            or history_item_ids is None
            or history_masks is None
        ):
            return None, None

        history_item_ids = history_item_ids.to(dtype=torch.long)
        history_masks = history_masks.to(dtype=torch.bool)
        clamped_item_ids = history_item_ids.clamp(
            min=0,
            max=max(int(self.item_popularity_bucket_ids.shape[0]) - 1, 0),
        )
        history_bucket_ids = self.item_popularity_bucket_ids[clamped_item_ids]
        if self.popularity_padding_bucket_id is None:
            raise ValueError("popularity_padding_bucket_id is not initialized.")
        padding_bucket_id = torch.full_like(history_bucket_ids, self.popularity_padding_bucket_id)
        history_bucket_ids = torch.where(
            history_masks & history_item_ids.ge(0),
            history_bucket_ids,
            padding_bucket_id,
        )

        history_popularity_embeds = self.popularity_embedding(history_bucket_ids)
        valid_mask = history_masks & history_item_ids.ge(0)
        valid_mask_f = valid_mask.unsqueeze(-1).to(dtype=history_popularity_embeds.dtype)
        denom = valid_mask_f.sum(dim=1).clamp_min(1.0)
        pooled_popularity_embed = (history_popularity_embeds * valid_mask_f).sum(dim=1) / denom
        no_valid_history = ~valid_mask.any(dim=1)
        if no_valid_history.any():
            pooled_popularity_embed = pooled_popularity_embed.clone()
            pooled_popularity_embed[no_valid_history] = 0
        return history_popularity_embeds, pooled_popularity_embed

    def _project_condition_branch(
        self,
        projector: ConditionBranchProjector | None,
        *,
        history_embeds: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        pooled_embed: torch.Tensor | None,
        target_embed: torch.Tensor | None,
        target_mask: torch.Tensor | None = None,
    ) -> ConditionProjectorOutput:
        if projector is None:
            return ConditionProjectorOutput(tokens=None, attention_mask=None)
        return projector(
            history_embeds=history_embeds,
            history_mask=history_mask,
            pooled_embed=pooled_embed,
            target_embed=target_embed,
            target_mask=target_mask,
        )

    def _apply_conditioning_dropout(
        self,
        branch_name: str,
        branch_output: ConditionProjectorOutput,
    ) -> ConditionProjectorOutput:
        if not self.training or branch_output.tokens is None or branch_output.attention_mask is None:
            return branch_output

        tokens = branch_output.tokens
        attention_mask = branch_output.attention_mask
        mask_changed = False

        branch_dropout_prob = self.branch_dropout_probs.get(branch_name, 0.0)
        if branch_dropout_prob > 0.0:
            sample_drop_mask = torch.rand(
                attention_mask.shape[0],
                device=attention_mask.device,
            ) < branch_dropout_prob
            if sample_drop_mask.any():
                attention_mask = attention_mask.clone()
                attention_mask[sample_drop_mask] = False
                mask_changed = True

        history_token_count = int(branch_output.history_token_count)
        history_keep_rate = self.history_token_keep_rates.get(branch_name, 1.0)
        if history_token_count > 0 and history_keep_rate < 1.0:
            history_mask = attention_mask[:, :history_token_count]
            if history_mask.any():
                sampled_keep_mask = torch.rand(
                    history_mask.shape,
                    device=history_mask.device,
                ) < history_keep_rate
                dropped_history_mask = history_mask & sampled_keep_mask

                if branch_output.pooled_token_count == 0:
                    empty_rows = history_mask.any(dim=1) & ~dropped_history_mask.any(dim=1)
                    if empty_rows.any():
                        dropped_history_mask = dropped_history_mask.clone()
                        for row_idx in torch.nonzero(empty_rows, as_tuple=False).flatten():
                            valid_positions = torch.nonzero(history_mask[row_idx], as_tuple=False).flatten()
                            selected_position = valid_positions[
                                torch.randint(valid_positions.numel(), (1,), device=history_mask.device)
                            ]
                            dropped_history_mask[row_idx, selected_position] = True

                attention_mask = attention_mask.clone()
                attention_mask[:, :history_token_count] = dropped_history_mask
                mask_changed = True

                if not self.keep_pooled_condition_tokens and branch_output.pooled_token_count > 0:
                    pooled_start = history_token_count
                    pooled_end = pooled_start + int(branch_output.pooled_token_count)
                    attention_mask[:, pooled_start:pooled_end] = dropped_history_mask.any(dim=1, keepdim=True)

        if not mask_changed:
            return branch_output

        masked_tokens = tokens.clone()
        masked_tokens = masked_tokens.masked_fill(~attention_mask.unsqueeze(-1), 0.0)
        return ConditionProjectorOutput(
            tokens=masked_tokens,
            attention_mask=attention_mask,
            history_token_count=branch_output.history_token_count,
            pooled_token_count=branch_output.pooled_token_count,
            target_token_count=branch_output.target_token_count,
        )

    def _extract_target_positions(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        token_codebook_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, _ = input_ids.shape
        positions = torch.zeros(
            (batch_size, self.code_len),
            dtype=torch.long,
            device=input_ids.device,
        )

        for row_idx in range(batch_size):
            if labels is not None:
                row_positions = torch.nonzero(labels[row_idx] != -100, as_tuple=False).flatten()
            else:
                row_positions = torch.nonzero(
                    (input_ids[row_idx] == self.mask_token_id) & attention_mask[row_idx].bool(),
                    as_tuple=False,
                ).flatten()

            if row_positions.numel() != self.code_len:
                raise ValueError(
                    "Unable to locate target positions for diffusion latent slots. "
                    f"Expected {self.code_len}, got {row_positions.numel()}."
                )

            if token_codebook_ids is not None:
                row_codebooks = token_codebook_ids[row_idx, row_positions]
                sort_index = torch.argsort(row_codebooks)
                row_positions = row_positions[sort_index]
            positions[row_idx] = row_positions
        return positions

    def _inject_noisy_target_latents(
        self,
        *,
        hidden_states: torch.Tensor,
        noisy_target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_target_latents.shape[0]
        latent_slots = noisy_target_latents.reshape(batch_size, self.code_len, self.latent_slot_dim)
        latent_tokens = self.target_latent_in(latent_slots)

        timestep_tokens = self.timestep_proj(self.timestep_embedder(timesteps)).unsqueeze(1)
        slot_indices = torch.arange(
            self.code_len,
            device=hidden_states.device,
            dtype=torch.long,
        ).unsqueeze(0)
        latent_tokens = latent_tokens + timestep_tokens + self.target_slot_embed(slot_indices)
        latent_tokens = latent_tokens.to(dtype=hidden_states.dtype)

        batch_index = torch.arange(batch_size, device=hidden_states.device, dtype=torch.long).unsqueeze(1)
        hidden_states[batch_index, target_positions] = hidden_states[batch_index, target_positions] + latent_tokens
        return hidden_states

    def _encode_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_slot_ids: torch.Tensor | None = None,
        token_codebook_ids: torch.Tensor | None = None,
        noisy_target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )
        if noisy_target_latents.ndim != 2 or noisy_target_latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"`noisy_target_latents` must have shape [B, {self.latent_dim}], got {tuple(noisy_target_latents.shape)}."
            )
        if noisy_target_latents.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between input_ids and noisy_target_latents: "
                f"{batch_size} vs {noisy_target_latents.shape[0]}."
            )
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(
                f"`timesteps` must have shape [B], got {tuple(timesteps.shape)}."
            )

        positions = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        x = self.token_embed(input_ids) + self.position_embed(positions)

        if token_slot_ids is not None:
            slot_ids = token_slot_ids.to(device=input_ids.device, dtype=torch.long) + 1
            slot_ids = slot_ids.clamp(min=0, max=self.max_history_len + 1)
            x = x + self.slot_embed(slot_ids)

        if token_codebook_ids is not None:
            codebook_ids = token_codebook_ids.to(device=input_ids.device, dtype=torch.long) + 1
            codebook_ids = codebook_ids.clamp(min=0, max=self.code_len + 1)
            x = x + self.codebook_embed(codebook_ids)

        target_positions = self._extract_target_positions(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            token_codebook_ids=token_codebook_ids,
        )
        x = self._inject_noisy_target_latents(
            hidden_states=x,
            noisy_target_latents=noisy_target_latents,
            timesteps=timesteps,
            target_positions=target_positions,
        )

        x = self.embed_dropout(self.input_norm(x))
        x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x, target_positions

    def _collect_target_hidden(
        self,
        hidden_states: torch.Tensor,
        target_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        batch_index = torch.arange(batch_size, device=hidden_states.device, dtype=torch.long).unsqueeze(1)
        return hidden_states[batch_index, target_positions]

    def _predict_x0_from_epsilon(
        self,
        *,
        noisy_target_latents: torch.Tensor,
        eps_prediction: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            device=noisy_target_latents.device,
            dtype=noisy_target_latents.dtype,
        )
        alpha_bar = alphas_cumprod[timesteps].unsqueeze(1)
        sqrt_alpha_bar = alpha_bar.sqrt()
        sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()
        return (noisy_target_latents - sqrt_one_minus_alpha_bar * eps_prediction) / sqrt_alpha_bar.clamp_min(1e-6)

    def _build_aux_queries(
        self,
        branch_output: ConditionProjectorOutput,
        head: nn.Linear | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if head is None or branch_output.tokens is None or branch_output.attention_mask is None:
            return None, None

        tokens = branch_output.tokens
        attention_mask = branch_output.attention_mask.to(device=tokens.device, dtype=torch.bool)
        pooled_query = None
        history_query = None

        history_end = int(branch_output.history_token_count)
        if branch_output.history_token_count > 0:
            history_tokens = tokens[:, :history_end]
            history_mask = attention_mask[:, :history_end]
            history_weight = history_mask.unsqueeze(-1).to(dtype=tokens.dtype)
            history_summary = (
                (history_tokens * history_weight).sum(dim=1)
                / history_weight.sum(dim=1).clamp_min(1.0)
            )
            history_query = head(history_summary)

        pooled_start = int(branch_output.history_token_count)
        pooled_end = pooled_start + int(branch_output.pooled_token_count)
        if branch_output.pooled_token_count > 0:
            pooled_tokens = tokens[:, pooled_start:pooled_end]
            pooled_mask = attention_mask[:, pooled_start:pooled_end]
            pooled_weight = pooled_mask.unsqueeze(-1).to(dtype=tokens.dtype)
            pooled_summary = (
                (pooled_tokens * pooled_weight).sum(dim=1)
                / pooled_weight.sum(dim=1).clamp_min(1.0)
            )
            pooled_query = head(pooled_summary)

        return pooled_query, history_query

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        noisy_target_latents: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
        token_slot_ids: torch.Tensor | None = None,
        token_codebook_ids: torch.Tensor | None = None,
        history_masks: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_item_ids: torch.Tensor | None = None,
        target_item_ids: torch.Tensor | None = None,
        history_text_embeds: torch.Tensor | None = None,
        target_text_embed: torch.Tensor | None = None,
        pooled_text_embed: torch.Tensor | None = None,
        history_image_embeds: torch.Tensor | None = None,
        target_image_embed: torch.Tensor | None = None,
        pooled_image_embed: torch.Tensor | None = None,
        history_cf_embeds: torch.Tensor | None = None,
        target_cf_embed: torch.Tensor | None = None,
        pooled_cf_embed: torch.Tensor | None = None,
        disable_popularity_condition: bool = False,
        **_: dict,
    ) -> HybridDiffusionOutput:
        history_masks = history_masks if history_masks is not None else history_mask
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if history_masks is not None:
            history_masks = history_masks.to(device=input_ids.device, dtype=torch.bool)

        hidden_states, target_positions = self._encode_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_slot_ids=token_slot_ids,
            token_codebook_ids=token_codebook_ids,
            noisy_target_latents=noisy_target_latents,
            timesteps=timesteps,
            labels=labels,
        )

        text_branch = self._apply_conditioning_dropout(
            "text",
            self._project_condition_branch(
                self.text_condition_projector,
                history_embeds=history_text_embeds,
                history_mask=history_masks,
                pooled_embed=pooled_text_embed,
                target_embed=target_text_embed,
            ),
        )
        image_branch = self._apply_conditioning_dropout(
            "image",
            self._project_condition_branch(
                self.image_condition_projector,
                history_embeds=history_image_embeds,
                history_mask=history_masks,
                pooled_embed=pooled_image_embed,
                target_embed=target_image_embed,
            ),
        )
        cf_branch = self._apply_conditioning_dropout(
            "cf",
            self._project_condition_branch(
                self.cf_condition_projector,
                history_embeds=history_cf_embeds,
                history_mask=history_masks,
                pooled_embed=pooled_cf_embed,
                target_embed=target_cf_embed,
            ),
        )
        popularity_branch = ConditionProjectorOutput(tokens=None, attention_mask=None)
        if not disable_popularity_condition and any(self.layer_schedule.popularity):
            history_popularity_embeds, pooled_popularity_embed = self._build_popularity_branch_inputs(
                history_item_ids=history_item_ids,
                history_masks=history_masks,
            )
            popularity_branch = self._apply_conditioning_dropout(
                "popularity",
                self._project_condition_branch(
                    self.popularity_condition_projector,
                    history_embeds=history_popularity_embeds,
                    history_mask=history_masks,
                    pooled_embed=pooled_popularity_embed,
                    target_embed=None,
                ),
            )

        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                text_tokens=text_branch.tokens if self.layer_schedule.text[layer_idx] else None,
                text_mask=text_branch.attention_mask if self.layer_schedule.text[layer_idx] else None,
                image_tokens=image_branch.tokens if self.layer_schedule.image[layer_idx] else None,
                image_mask=image_branch.attention_mask if self.layer_schedule.image[layer_idx] else None,
                cf_tokens=cf_branch.tokens if self.layer_schedule.cf[layer_idx] else None,
                cf_mask=cf_branch.attention_mask if self.layer_schedule.cf[layer_idx] else None,
                popularity_tokens=(
                    popularity_branch.tokens if self.layer_schedule.popularity[layer_idx] else None
                ),
                popularity_mask=(
                    popularity_branch.attention_mask
                    if self.layer_schedule.popularity[layer_idx]
                    else None
                ),
            )

        hidden_states = self.final_norm(hidden_states)
        target_hidden = self._collect_target_hidden(hidden_states, target_positions)
        prediction = self.target_latent_out(target_hidden).reshape(input_ids.shape[0], self.latent_dim)

        if self.prediction_type == "epsilon":
            denoised = self._predict_x0_from_epsilon(
                noisy_target_latents=noisy_target_latents,
                eps_prediction=prediction,
                timesteps=timesteps,
            )
        else:
            denoised = prediction

        text_aux_query, text_history_aux_query = self._build_aux_queries(text_branch, self.text_aux_head)
        image_aux_query, image_history_aux_query = self._build_aux_queries(image_branch, self.image_aux_head)

        return HybridDiffusionOutput(
            prediction=prediction,
            denoised_latents=denoised,
            target_positions=target_positions,
            hidden_states=hidden_states,
            text_aux_query=text_aux_query,
            image_aux_query=image_aux_query,
            text_history_aux_query=text_history_aux_query,
            image_history_aux_query=image_history_aux_query,
        )

    def compute_losses(
        self,
        *,
        output: HybridDiffusionOutput,
        target_latents: torch.Tensor,
        noise: torch.Tensor,
        diffusion_loss_weight: float = 1.0,
        ranking_loss_weight: float = 0.0,
        ranking_temperature: float = 0.07,
        text_auxiliary_loss_weight: float = 0.0,
        image_auxiliary_loss_weight: float = 0.0,
        auxiliary_ranking_temperature: float | None = None,
        target_item_ids: torch.Tensor | None = None,
        item_embedding_table: torch.Tensor | None = None,
        ranking_sample_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        diffusion_target = noise if self.prediction_type == "epsilon" else target_latents
        diffusion_loss = F.mse_loss(output.prediction.float(), diffusion_target.float())
        total_loss = diffusion_loss * float(diffusion_loss_weight)
        item_table = None
        if item_embedding_table is not None:
            item_table = F.normalize(item_embedding_table.float(), dim=-1)

        ranking_loss = None
        if (
            float(ranking_loss_weight) > 0
            and target_item_ids is not None
            and item_embedding_table is not None
        ):
            pred_query = F.normalize(output.denoised_latents.float(), dim=-1)
            assert item_table is not None
            logits = pred_query @ item_table.t()
            logits = logits / max(float(ranking_temperature), 1e-6)
            per_sample_ranking_loss = F.cross_entropy(
                logits,
                target_item_ids.to(device=pred_query.device, dtype=torch.long),
                reduction="none",
            )
            if ranking_sample_weights is not None:
                sample_weights = ranking_sample_weights.to(
                    device=per_sample_ranking_loss.device,
                    dtype=per_sample_ranking_loss.dtype,
                )
                ranking_loss = (
                    per_sample_ranking_loss * sample_weights
                ).sum() / sample_weights.sum().clamp_min(1e-6)
            else:
                ranking_loss = per_sample_ranking_loss.mean()
            total_loss = total_loss + ranking_loss * float(ranking_loss_weight)

        aux_temperature = (
            float(auxiliary_ranking_temperature)
            if auxiliary_ranking_temperature is not None
            else float(ranking_temperature)
        )
        text_auxiliary_loss = None
        if (
            float(text_auxiliary_loss_weight) > 0
            and output.text_aux_query is not None
            and target_item_ids is not None
            and item_embedding_table is not None
        ):
            assert item_table is not None
            text_aux_components: list[torch.Tensor] = []

            text_query = F.normalize(output.text_aux_query.float(), dim=-1)
            text_logits = text_query @ item_table.t()
            text_logits = text_logits / max(aux_temperature, 1e-6)
            text_per_sample_loss = F.cross_entropy(
                text_logits,
                target_item_ids.to(device=text_query.device, dtype=torch.long),
                reduction="none",
            )
            if ranking_sample_weights is not None:
                text_sample_weights = ranking_sample_weights.to(
                    device=text_per_sample_loss.device,
                    dtype=text_per_sample_loss.dtype,
                )
                text_aux_components.append(
                    (text_per_sample_loss * text_sample_weights).sum()
                    / text_sample_weights.sum().clamp_min(1e-6)
                )
            else:
                text_aux_components.append(text_per_sample_loss.mean())

            if output.text_history_aux_query is not None:
                text_history_query = F.normalize(output.text_history_aux_query.float(), dim=-1)
                text_history_logits = text_history_query @ item_table.t()
                text_history_logits = text_history_logits / max(aux_temperature, 1e-6)
                text_history_per_sample_loss = F.cross_entropy(
                    text_history_logits,
                    target_item_ids.to(device=text_history_query.device, dtype=torch.long),
                    reduction="none",
                )
                if ranking_sample_weights is not None:
                    text_history_sample_weights = ranking_sample_weights.to(
                        device=text_history_per_sample_loss.device,
                        dtype=text_history_per_sample_loss.dtype,
                    )
                    text_aux_components.append(
                        (text_history_per_sample_loss * text_history_sample_weights).sum()
                        / text_history_sample_weights.sum().clamp_min(1e-6)
                    )
                else:
                    text_aux_components.append(text_history_per_sample_loss.mean())

            text_auxiliary_loss = torch.stack(text_aux_components).mean()
            total_loss = total_loss + text_auxiliary_loss * float(text_auxiliary_loss_weight)

        image_auxiliary_loss = None
        if (
            float(image_auxiliary_loss_weight) > 0
            and output.image_aux_query is not None
            and target_item_ids is not None
            and item_embedding_table is not None
        ):
            assert item_table is not None
            image_aux_components: list[torch.Tensor] = []

            image_query = F.normalize(output.image_aux_query.float(), dim=-1)
            image_logits = image_query @ item_table.t()
            image_logits = image_logits / max(aux_temperature, 1e-6)
            image_per_sample_loss = F.cross_entropy(
                image_logits,
                target_item_ids.to(device=image_query.device, dtype=torch.long),
                reduction="none",
            )
            if ranking_sample_weights is not None:
                image_sample_weights = ranking_sample_weights.to(
                    device=image_per_sample_loss.device,
                    dtype=image_per_sample_loss.dtype,
                )
                image_aux_components.append(
                    (image_per_sample_loss * image_sample_weights).sum()
                    / image_sample_weights.sum().clamp_min(1e-6)
                )
            else:
                image_aux_components.append(image_per_sample_loss.mean())

            if output.image_history_aux_query is not None:
                image_history_query = F.normalize(output.image_history_aux_query.float(), dim=-1)
                image_history_logits = image_history_query @ item_table.t()
                image_history_logits = image_history_logits / max(aux_temperature, 1e-6)
                image_history_per_sample_loss = F.cross_entropy(
                    image_history_logits,
                    target_item_ids.to(device=image_history_query.device, dtype=torch.long),
                    reduction="none",
                )
                if ranking_sample_weights is not None:
                    image_history_sample_weights = ranking_sample_weights.to(
                        device=image_history_per_sample_loss.device,
                        dtype=image_history_per_sample_loss.dtype,
                    )
                    image_aux_components.append(
                        (image_history_per_sample_loss * image_history_sample_weights).sum()
                        / image_history_sample_weights.sum().clamp_min(1e-6)
                    )
                else:
                    image_aux_components.append(image_history_per_sample_loss.mean())

            image_auxiliary_loss = torch.stack(image_aux_components).mean()
            total_loss = total_loss + image_auxiliary_loss * float(image_auxiliary_loss_weight)

        payload = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
        }
        if ranking_loss is not None:
            payload["ranking_loss"] = ranking_loss
        if text_auxiliary_loss is not None:
            payload["text_auxiliary_loss"] = text_auxiliary_loss
        if image_auxiliary_loss is not None:
            payload["image_auxiliary_loss"] = image_auxiliary_loss
        return payload

    def prepare_training_inputs(
        self,
        *,
        target_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = target_latents.shape[0]
        device = target_latents.device
        noise = torch.randn_like(target_latents)
        timesteps = torch.randint(
            0,
            int(self.noise_scheduler.config.num_train_timesteps),
            (batch_size,),
            device=device,
            dtype=torch.long,
        )
        noisy_target_latents = self.noise_scheduler.add_noise(target_latents, noise, timesteps)
        return noisy_target_latents, noise, timesteps

    @torch.no_grad()
    def sample_latents(
        self,
        batch: SemanticIdBatch | dict[str, torch.Tensor],
        *,
        num_inference_steps: int = 50,
        disable_popularity_condition: bool = False,
    ) -> torch.Tensor:
        if isinstance(batch, dict):
            batch = SemanticIdBatch.from_dict(batch)

        device = self.token_embed.weight.device
        dtype = self.token_embed.weight.dtype
        batch_size = int(batch.input_ids.shape[0])
        x_t = torch.randn(batch_size, self.latent_dim, device=device, dtype=dtype)

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=int(self.noise_scheduler.config.num_train_timesteps),
            beta_schedule=str(self.noise_scheduler.config.beta_schedule),
            prediction_type=self.prediction_type,
            clip_sample=False,
        )
        ddim_scheduler.set_timesteps(int(num_inference_steps))

        for t in ddim_scheduler.timesteps:
            timesteps = torch.full(
                (batch_size,),
                int(t),
                device=device,
                dtype=torch.long,
            )
            output = self.forward(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                labels=batch.labels,
                history_item_ids=batch.history_item_ids,
                target_item_ids=batch.target_item_ids,
                token_slot_ids=batch.token_slot_ids,
                token_codebook_ids=batch.token_codebook_ids,
                history_masks=batch.history_masks,
                history_text_embeds=batch.history_text_embeds,
                target_text_embed=batch.target_text_embed,
                pooled_text_embed=batch.pooled_text_embed,
                history_image_embeds=batch.history_image_embeds,
                target_image_embed=batch.target_image_embed,
                pooled_image_embed=batch.pooled_image_embed,
                history_cf_embeds=batch.history_cf_embeds,
                target_cf_embed=batch.target_cf_embed,
                pooled_cf_embed=batch.pooled_cf_embed,
                disable_popularity_condition=disable_popularity_condition,
                noisy_target_latents=x_t,
                timesteps=timesteps,
            )
            x_t = ddim_scheduler.step(output.prediction, t, x_t).prev_sample.to(dtype=dtype)

        return x_t
