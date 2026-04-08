from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class SemanticTokenLayout:
    code_len: int
    vocab_sizes: list[int]
    pad_token_id: int = 0
    mask_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    eos_token_id: int = 4

    @property
    def num_special_tokens(self) -> int:
        return 5

    @property
    def semantic_offsets(self) -> list[int]:
        offsets = []
        running = self.num_special_tokens
        for vocab_size in self.vocab_sizes:
            offsets.append(running)
            running += int(vocab_size)
        return offsets

    @property
    def vocab_size(self) -> int:
        return self.num_special_tokens + int(sum(self.vocab_sizes))

    @property
    def max_history_seq_len(self) -> int:
        return 1 + self.code_len + 1

    def encode_code(self, codebook_idx: int, code_value: int) -> int:
        if codebook_idx < 0 or codebook_idx >= self.code_len:
            raise IndexError(f"Invalid codebook_idx {codebook_idx} for code_len={self.code_len}.")
        if code_value < 0 or code_value >= self.vocab_sizes[codebook_idx]:
            raise ValueError(
                f"Invalid code value {code_value} for codebook_idx={codebook_idx} "
                f"with vocab_size={self.vocab_sizes[codebook_idx]}."
            )
        return self.semantic_offsets[codebook_idx] + int(code_value)

    def encode_codes(self, codes: Sequence[int]) -> list[int]:
        if len(codes) != self.code_len:
            raise ValueError(f"Expected {self.code_len} semantic codes, got {len(codes)}.")
        return [self.encode_code(codebook_idx, int(code_value)) for codebook_idx, code_value in enumerate(codes)]


def load_item_to_code(path: str | Path) -> dict[str, list[int]]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    return {
        str(item_id): [int(value) for value in code]
        for item_id, code in payload.items()
    }


def infer_vocab_sizes(item_to_code: dict[str, list[int]]) -> list[int]:
    iterator = iter(item_to_code.values())
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("item_to_code is empty.") from exc

    code_len = len(first)
    maxima = [int(value) for value in first]
    for code in iterator:
        if len(code) != code_len:
            raise ValueError("Inconsistent semantic code length across items.")
        for idx, value in enumerate(code):
            maxima[idx] = max(maxima[idx], int(value))
    return [value + 1 for value in maxima]


def build_layout_from_item_to_code(
    item_to_code: dict[str, list[int]],
    *,
    pad_token_id: int = 0,
    mask_token_id: int = 1,
    cls_token_id: int = 2,
    sep_token_id: int = 3,
    eos_token_id: int = 4,
) -> SemanticTokenLayout:
    vocab_sizes = infer_vocab_sizes(item_to_code)
    code_len = len(next(iter(item_to_code.values())))
    return SemanticTokenLayout(
        code_len=code_len,
        vocab_sizes=vocab_sizes,
        pad_token_id=pad_token_id,
        mask_token_id=mask_token_id,
        cls_token_id=cls_token_id,
        sep_token_id=sep_token_id,
        eos_token_id=eos_token_id,
    )
