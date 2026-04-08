from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm, use_fused_attn

from genrec.contracts import SemanticIdBatch
from models.rdt.blocks import CrossAttention

from .condition_projector import ConditionBranchProjector


class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention with an optional padding mask."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = RmsNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}.")

        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"Expected attention_mask shape {(batch_size, seq_len)}, "
                    f"got {tuple(attention_mask.shape)}."
                )
            attn_mask = attention_mask[:, None, None, :].expand(-1, -1, seq_len, -1)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            scores = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask.logical_not(), float("-inf"))
            scores = scores.softmax(dim=-1)
            if self.attn_drop.p > 0:
                scores = self.attn_drop(scores)
            x = scores @ v

        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x


class GenRecDiTBlock(nn.Module):
    """DiT-style block for semantic-token recommendation with RDT-style multimodal injection."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RmsNorm(hidden_size, eps=1e-6)
        self.self_attn = MaskedSelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=dropout,
            proj_drop=dropout,
            norm_layer=RmsNorm,
        )
        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        self.text_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=dropout,
            proj_drop=dropout,
            norm_layer=RmsNorm,
        )
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)
        self.image_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=dropout,
            proj_drop=dropout,
            norm_layer=RmsNorm,
        )
        self.norm4 = RmsNorm(hidden_size, eps=1e-6)
        self.cf_cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=dropout,
            proj_drop=dropout,
            norm_layer=RmsNorm,
        )
        self.norm5 = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 4,
            act_layer=approx_gelu,
            drop=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
        image_tokens: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        cf_tokens: torch.Tensor | None = None,
        cf_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.text_cross_attn(self.norm2(x), text_tokens, text_mask)
        x = x + self.image_cross_attn(self.norm3(x), image_tokens, image_mask)
        x = x + self.cf_cross_attn(self.norm4(x), cf_tokens, cf_mask)
        x = x + self.ffn(self.norm5(x))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x


@dataclass
class GenRecForwardOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    loss: torch.Tensor | None = None
    masked_token_accuracy: torch.Tensor | None = None


class GenRecDiTRunner(nn.Module):
    """
    Semantic-ID recommendation backbone using a DiT-style token encoder.

    The sequence branch consumes semantic-ID tokens built from user history.
    RDT-style cross-attention branches inject:
    - text conditions
    - image conditions
    - CF conditions
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        vocab_sizes: list[int],
        max_seq_len: int,
        max_history_len: int,
        hidden_size: int = 1024,
        depth: int = 12,
        num_heads: int = 16,
        num_special_tokens: int = 5,
        pad_token_id: int = 0,
        mask_token_id: int = 1,
        label_smoothing: float = 0.0,
        restrict_target_vocab: bool = True,
        text_cond_dim: int | None = None,
        image_cond_dim: int | None = None,
        cf_cond_dim: int | None = None,
        use_text_history: bool = True,
        use_text_pooled: bool = True,
        use_text_target: bool = False,
        use_image_history: bool = True,
        use_image_pooled: bool = True,
        use_image_target: bool = False,
        use_cf_history: bool = True,
        use_cf_pooled: bool = True,
        use_cf_target: bool = False,
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
        self.label_smoothing = float(label_smoothing)
        self.restrict_target_vocab = bool(restrict_target_vocab)

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

        self.blocks = nn.ModuleList(
            [GenRecDiTBlock(self.hidden_size, self.num_heads, dropout=dropout) for _ in range(self.depth)]
        )
        self.final_norm = RmsNorm(self.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        self.text_condition_projector = None
        if text_cond_dim is not None:
            self.text_condition_projector = ConditionBranchProjector(
                input_dim=int(text_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_text_history,
                use_pooled=use_text_pooled,
                use_target=use_text_target,
                dropout=dropout,
            )

        self.image_condition_projector = None
        if image_cond_dim is not None:
            self.image_condition_projector = ConditionBranchProjector(
                input_dim=int(image_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_image_history,
                use_pooled=use_image_pooled,
                use_target=use_image_target,
                dropout=dropout,
            )

        self.cf_condition_projector = None
        if cf_cond_dim is not None:
            self.cf_condition_projector = ConditionBranchProjector(
                input_dim=int(cf_cond_dim),
                hidden_size=self.hidden_size,
                max_history_len=self.max_history_len,
                use_history=use_cf_history,
                use_pooled=use_cf_pooled,
                use_target=use_cf_target,
                dropout=dropout,
            )

        semantic_offsets = []
        running = self.num_special_tokens
        for size in self.vocab_sizes:
            semantic_offsets.append(running)
            running += int(size)
        self.register_buffer(
            "semantic_offsets",
            torch.tensor(semantic_offsets, dtype=torch.long),
            persistent=True,
        )

        codebook_vocab_mask = torch.zeros(self.code_len, self.vocab_size, dtype=torch.bool)
        for codebook_idx, (offset, size) in enumerate(zip(semantic_offsets, self.vocab_sizes)):
            codebook_vocab_mask[codebook_idx, offset : offset + size] = True
        self.register_buffer(
            "codebook_vocab_mask",
            codebook_vocab_mask,
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, std=0.02)
        if self.pad_token_id >= 0:
            with torch.no_grad():
                self.token_embed.weight[self.pad_token_id].zero_()
        nn.init.normal_(self.position_embed.weight, std=0.02)
        nn.init.normal_(self.slot_embed.weight, std=0.02)
        nn.init.normal_(self.codebook_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @classmethod
    def from_batch_metadata(
        cls,
        *,
        manifest: dict,
        max_history_len: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        text_cond_dim: int | None = None,
        image_cond_dim: int | None = None,
        cf_cond_dim: int | None = None,
        label_smoothing: float = 0.0,
        restrict_target_vocab: bool = True,
        use_text_history: bool = True,
        use_text_pooled: bool = True,
        use_text_target: bool = False,
        use_image_history: bool = True,
        use_image_pooled: bool = True,
        use_image_target: bool = False,
        use_cf_history: bool = True,
        use_cf_pooled: bool = True,
        use_cf_target: bool = False,
        dropout: float = 0.0,
    ) -> "GenRecDiTRunner":
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
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            num_special_tokens=len(special_tokens) or 5,
            pad_token_id=int(special_tokens.get("pad_token_id", 0)),
            mask_token_id=int(special_tokens.get("mask_token_id", 1)),
            label_smoothing=label_smoothing,
            restrict_target_vocab=restrict_target_vocab,
            text_cond_dim=text_cond_dim,
            image_cond_dim=image_cond_dim,
            cf_cond_dim=cf_cond_dim,
            use_text_history=use_text_history,
            use_text_pooled=use_text_pooled,
            use_text_target=use_text_target,
            use_image_history=use_image_history,
            use_image_pooled=use_image_pooled,
            use_image_target=use_image_target,
            use_cf_history=use_cf_history,
            use_cf_pooled=use_cf_pooled,
            use_cf_target=use_cf_target,
            dropout=dropout,
        )

    def code_value_to_token_id(self, codebook_idx: int, code_value: int) -> int:
        return int(self.semantic_offsets[codebook_idx].item()) + int(code_value)

    def token_id_to_code_value(self, codebook_idx: int, token_id: int) -> int:
        return int(token_id) - int(self.semantic_offsets[codebook_idx].item())

    def _encode_sequence(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_slot_ids: torch.Tensor | None = None,
        token_codebook_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
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

        x = self.embed_dropout(self.input_norm(x))
        x = x * attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        return x

    def _project_condition_branch(
        self,
        projector: ConditionBranchProjector | None,
        *,
        history_embeds: torch.Tensor | None,
        history_mask: torch.Tensor | None,
        pooled_embed: torch.Tensor | None,
        target_embed: torch.Tensor | None,
        target_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if projector is None:
            return None, None
        output = projector(
            history_embeds=history_embeds,
            history_mask=history_mask,
            pooled_embed=pooled_embed,
            target_embed=target_embed,
            target_mask=target_mask,
        )
        return output.tokens, output.attention_mask

    def constrain_logits_to_codebooks(
        self,
        logits: torch.Tensor,
        token_codebook_ids: torch.Tensor,
        *,
        positions_mask: torch.Tensor | None = None,
        allowed_token_ids: list[list[int]] | None = None,
    ) -> torch.Tensor:
        constrained = logits.clone()
        for codebook_idx in range(self.code_len):
            active_positions = token_codebook_ids == codebook_idx
            if positions_mask is not None:
                active_positions = active_positions & positions_mask
            if not active_positions.any():
                continue

            allowed_vocab = self.codebook_vocab_mask[codebook_idx].to(device=logits.device)
            constrained[active_positions] = constrained[active_positions].masked_fill(
                (~allowed_vocab).unsqueeze(0),
                -1e9,
            )

        if allowed_token_ids is not None:
            if len(allowed_token_ids) != logits.shape[0]:
                raise ValueError(
                    f"allowed_token_ids length {len(allowed_token_ids)} does not match batch size {logits.shape[0]}."
                )
            for batch_idx, row_allowed in enumerate(allowed_token_ids):
                if not row_allowed:
                    continue
                row_mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
                row_mask[torch.tensor(row_allowed, device=logits.device, dtype=torch.long)] = True
                constrained[batch_idx] = constrained[batch_idx].masked_fill(~row_mask, -1e9)
        return constrained

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        token_slot_ids: torch.Tensor | None = None,
        token_codebook_ids: torch.Tensor | None = None,
        history_masks: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        history_text_embeds: torch.Tensor | None = None,
        target_text_embed: torch.Tensor | None = None,
        pooled_text_embed: torch.Tensor | None = None,
        history_image_embeds: torch.Tensor | None = None,
        target_image_embed: torch.Tensor | None = None,
        pooled_image_embed: torch.Tensor | None = None,
        history_cf_embeds: torch.Tensor | None = None,
        target_cf_embed: torch.Tensor | None = None,
        pooled_cf_embed: torch.Tensor | None = None,
        **_: dict,
    ) -> GenRecForwardOutput:
        history_masks = history_masks if history_masks is not None else history_mask
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        if history_masks is not None:
            history_masks = history_masks.to(device=input_ids.device, dtype=torch.bool)

        hidden_states = self._encode_sequence(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_slot_ids=token_slot_ids,
            token_codebook_ids=token_codebook_ids,
        )

        text_tokens, text_mask = self._project_condition_branch(
            self.text_condition_projector,
            history_embeds=history_text_embeds,
            history_mask=history_masks,
            pooled_embed=pooled_text_embed,
            target_embed=target_text_embed,
        )
        image_tokens, image_mask = self._project_condition_branch(
            self.image_condition_projector,
            history_embeds=history_image_embeds,
            history_mask=history_masks,
            pooled_embed=pooled_image_embed,
            target_embed=target_image_embed,
        )
        cf_tokens, cf_mask = self._project_condition_branch(
            self.cf_condition_projector,
            history_embeds=history_cf_embeds,
            history_mask=history_masks,
            pooled_embed=pooled_cf_embed,
            target_embed=target_cf_embed,
        )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                text_tokens=text_tokens,
                text_mask=text_mask,
                image_tokens=image_tokens,
                image_mask=image_mask,
                cf_tokens=cf_tokens,
                cf_mask=cf_mask,
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        masked_token_accuracy = None
        if labels is not None:
            positions_mask = labels != -100
            masked_logits = logits
            if self.restrict_target_vocab and token_codebook_ids is not None:
                masked_logits = self.constrain_logits_to_codebooks(
                    logits,
                    token_codebook_ids=token_codebook_ids,
                    positions_mask=positions_mask,
                )

            loss = F.cross_entropy(
                masked_logits.reshape(-1, masked_logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=self.label_smoothing,
            )

            with torch.no_grad():
                predictions = masked_logits.argmax(dim=-1)
                correct = ((predictions == labels) & positions_mask).sum()
                total = positions_mask.sum().clamp(min=1)
                masked_token_accuracy = correct.float() / total.float()

            logits = masked_logits

        return GenRecForwardOutput(
            logits=logits,
            hidden_states=hidden_states,
            loss=loss,
            masked_token_accuracy=masked_token_accuracy,
        )

    @torch.no_grad()
    def greedy_decode(
        self,
        batch: SemanticIdBatch | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch, dict):
            batch = SemanticIdBatch.from_dict(batch)

        input_ids = batch.input_ids.clone()
        attention_mask = batch.attention_mask
        token_codebook_ids = batch.token_codebook_ids
        if token_codebook_ids is None:
            raise ValueError("greedy_decode requires `token_codebook_ids`.")

        target_positions_per_row = []
        for row_idx in range(input_ids.shape[0]):
            row_positions = torch.nonzero(
                (input_ids[row_idx] == self.mask_token_id) & attention_mask[row_idx].bool(),
                as_tuple=False,
            ).flatten()
            target_positions_per_row.append(row_positions.tolist())

        predicted_token_ids = torch.full_like(input_ids, fill_value=-1)
        predicted_code_values = torch.full_like(input_ids, fill_value=-1)

        max_target_len = max((len(row) for row in target_positions_per_row), default=0)
        for decode_step in range(max_target_len):
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_slot_ids=batch.token_slot_ids,
                token_codebook_ids=token_codebook_ids,
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
            )

            for row_idx, row_positions in enumerate(target_positions_per_row):
                if decode_step >= len(row_positions):
                    continue
                position = row_positions[decode_step]
                codebook_idx = int(token_codebook_ids[row_idx, position].item())
                row_logits = outputs.logits[row_idx, position].unsqueeze(0)
                row_logits = self.constrain_logits_to_codebooks(
                    row_logits,
                    token_codebook_ids=torch.tensor(
                        [[codebook_idx]],
                        device=row_logits.device,
                        dtype=torch.long,
                    ),
                    positions_mask=torch.ones(
                        (1, 1),
                        device=row_logits.device,
                        dtype=torch.bool,
                    ),
                )[0]

                predicted_token_id = int(torch.argmax(row_logits).item())
                predicted_code_value = self.token_id_to_code_value(codebook_idx, predicted_token_id)
                input_ids[row_idx, position] = predicted_token_id
                predicted_token_ids[row_idx, position] = predicted_token_id
                predicted_code_values[row_idx, position] = predicted_code_value

        return {
            "decoded_input_ids": input_ids,
            "predicted_token_ids": predicted_token_ids,
            "predicted_code_values": predicted_code_values,
        }
