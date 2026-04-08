from __future__ import annotations

import json
from pathlib import Path

import torch

from genrec.contracts import SemanticIdBatch
from genrec.models import GenRecDiTRunner

from .prefix_trie import SemanticPrefixTrie


def load_code_to_items(path: str | Path) -> dict[tuple[int, ...], list[str]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {
        tuple(int(value) for value in key.split("-")): [str(item_id) for item_id in item_ids]
        for key, item_ids in payload.items()
    }


def build_code_to_items(item_to_code: dict[str, list[int]]) -> dict[tuple[int, ...], list[str]]:
    code_to_items: dict[tuple[int, ...], list[str]] = {}
    for item_id, code_sequence in item_to_code.items():
        code_to_items.setdefault(tuple(int(value) for value in code_sequence), []).append(str(item_id))
    return code_to_items


def decode_codes_to_items(
    predicted_codes: list[list[int]],
    code_to_items: dict[tuple[int, ...], list[str]],
) -> list[list[str]]:
    decoded_items: list[list[str]] = []
    for code_sequence in predicted_codes:
        decoded_items.append(code_to_items.get(tuple(int(value) for value in code_sequence), []))
    return decoded_items


@torch.no_grad()
def greedy_decode_with_prefix(
    model: GenRecDiTRunner,
    batch: SemanticIdBatch | dict[str, torch.Tensor],
    *,
    prefix_trie: SemanticPrefixTrie | None = None,
) -> dict[str, torch.Tensor | list[list[int]]]:
    if isinstance(batch, dict):
        batch = SemanticIdBatch.from_dict(batch)

    input_ids = batch.input_ids.clone()
    attention_mask = batch.attention_mask
    token_codebook_ids = batch.token_codebook_ids
    if token_codebook_ids is None:
        raise ValueError("greedy_decode_with_prefix requires `token_codebook_ids`.")

    masked_positions = []
    for row_idx in range(input_ids.shape[0]):
        row_positions = torch.nonzero(
            (input_ids[row_idx] == model.mask_token_id) & attention_mask[row_idx].bool(),
            as_tuple=False,
        ).flatten()
        masked_positions.append(row_positions.tolist())

    generated_codes: list[list[int]] = [[] for _ in range(input_ids.shape[0])]
    predicted_token_ids = torch.full_like(input_ids, fill_value=-1)

    max_steps = max((len(row) for row in masked_positions), default=0)
    for decode_step in range(max_steps):
        outputs = model(
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

        for row_idx, row_mask_positions in enumerate(masked_positions):
            if decode_step >= len(row_mask_positions):
                continue

            position = row_mask_positions[decode_step]
            codebook_idx = int(token_codebook_ids[row_idx, position].item())
            row_logits = outputs.logits[row_idx, position].unsqueeze(0)
            row_logits = model.constrain_logits_to_codebooks(
                row_logits,
                token_codebook_ids=torch.tensor(
                    [[codebook_idx]],
                    device=row_logits.device,
                    dtype=torch.long,
                ),
                positions_mask=torch.ones((1, 1), device=row_logits.device, dtype=torch.bool),
            )[0]

            if prefix_trie is not None:
                allowed_values = prefix_trie.allowed_next(generated_codes[row_idx])
                if allowed_values:
                    allowed_tokens = [model.code_value_to_token_id(codebook_idx, value) for value in allowed_values]
                    allowed_mask = torch.zeros_like(row_logits, dtype=torch.bool)
                    allowed_mask[torch.tensor(allowed_tokens, device=row_logits.device, dtype=torch.long)] = True
                    row_logits = row_logits.masked_fill(~allowed_mask, -1e9)

            predicted_token_id = int(torch.argmax(row_logits).item())
            predicted_code_value = model.token_id_to_code_value(codebook_idx, predicted_token_id)

            input_ids[row_idx, position] = predicted_token_id
            predicted_token_ids[row_idx, position] = predicted_token_id
            generated_codes[row_idx].append(predicted_code_value)

    return {
        "decoded_input_ids": input_ids,
        "predicted_token_ids": predicted_token_ids,
        "predicted_codes": generated_codes,
    }
