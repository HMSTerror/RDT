from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class SemanticIdBatch:
    """Common batch contract for future GenRec training/evaluation code."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    target_item_ids: Optional[torch.Tensor] = None
    target_item_latent: Optional[torch.Tensor] = None
    history_item_ids: Optional[torch.Tensor] = None
    history_masks: Optional[torch.Tensor] = None
    history_semantic_ids: Optional[torch.Tensor] = None
    target_semantic_ids: Optional[torch.Tensor] = None
    token_slot_ids: Optional[torch.Tensor] = None
    token_codebook_ids: Optional[torch.Tensor] = None
    seq_lengths: Optional[torch.Tensor] = None
    history_text_embeds: Optional[torch.Tensor] = None
    target_text_embed: Optional[torch.Tensor] = None
    pooled_text_embed: Optional[torch.Tensor] = None
    history_image_embeds: Optional[torch.Tensor] = None
    target_image_embed: Optional[torch.Tensor] = None
    pooled_image_embed: Optional[torch.Tensor] = None
    history_cf_embeds: Optional[torch.Tensor] = None
    target_cf_embed: Optional[torch.Tensor] = None
    pooled_cf_embed: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, batch: dict[str, Any]) -> "SemanticIdBatch":
        """Normalize dataset / collator aliases into a common batch contract."""
        return cls(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch.get("labels"),
            target_item_ids=batch.get("target_item_ids", batch.get("target_item_id")),
            target_item_latent=batch.get("target_item_latent"),
            history_item_ids=batch.get("history_item_ids"),
            history_masks=batch.get("history_masks", batch.get("history_mask")),
            history_semantic_ids=batch.get("history_semantic_ids"),
            target_semantic_ids=batch.get("target_semantic_ids"),
            token_slot_ids=batch.get("token_slot_ids"),
            token_codebook_ids=batch.get("token_codebook_ids"),
            seq_lengths=batch.get("seq_lengths", batch.get("seq_length")),
            history_text_embeds=batch.get("history_text_embeds"),
            target_text_embed=batch.get("target_text_embed"),
            pooled_text_embed=batch.get("pooled_text_embed"),
            history_image_embeds=batch.get("history_image_embeds"),
            target_image_embed=batch.get("target_image_embed"),
            pooled_image_embed=batch.get("pooled_image_embed"),
            history_cf_embeds=batch.get("history_cf_embeds"),
            target_cf_embed=batch.get("target_cf_embed"),
            pooled_cf_embed=batch.get("pooled_cf_embed"),
        )

    def to(self, device: torch.device | str) -> "SemanticIdBatch":
        """Move every tensor field to the requested device."""
        payload: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if torch.is_tensor(value):
                payload[field_name] = value.to(device=device)
            else:
                payload[field_name] = value
        return SemanticIdBatch(**payload)


@dataclass
class DecodeResult:
    """Common decode output contract for future beam/greedy inference."""

    sequence_ids: torch.Tensor
    item_ids: torch.Tensor
    scores: torch.Tensor
