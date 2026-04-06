from types import SimpleNamespace

import torch
import torch.nn as nn


class DummyTokenizer:
    """
    Offline-only tokenizer for smoke tests and minimal end-to-end validation.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = int(vocab_size)

    def __call__(
        self,
        texts,
        max_length,
        padding,
        truncation,
        return_attention_mask,
        add_special_tokens,
        return_tensors,
    ):
        if padding != "max_length":
            raise ValueError("DummyTokenizer only supports padding='max_length'.")
        if return_tensors != "pt":
            raise ValueError("DummyTokenizer only supports return_tensors='pt'.")

        batch_size = len(texts)
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

        for row, text in enumerate(texts):
            encoded = [((ord(ch) % (self.vocab_size - 1)) + 1) for ch in str(text)]
            if add_special_tokens:
                encoded = [1] + encoded + [2]
            if truncation:
                encoded = encoded[:max_length]

            length = min(len(encoded), max_length)
            if length > 0:
                input_ids[row, :length] = torch.tensor(encoded[:length], dtype=torch.long)
                if return_attention_mask:
                    attention_mask[row, :length] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class DummyTextEncoder(nn.Module):
    """
    Lightweight text encoder that mimics the T5 encoder output contract.
    """

    def __init__(self, vocab_size: int = 512, hidden_size: int = 48) -> None:
        super().__init__()
        self.config = SimpleNamespace(d_model=int(hidden_size))
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, input_ids, attention_mask):
        del attention_mask
        hidden = self.embed(input_ids)
        hidden = self.proj(hidden)
        return {"last_hidden_state": hidden}
