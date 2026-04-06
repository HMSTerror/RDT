# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn

from models.rdt.blocks import RDTBlock, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid


class RDT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        output_dim=128,
        horizon=1,
        hidden_size=1024,
        depth=28,
        num_heads=16,
        max_lang_cond_len=None,
        img_cond_len=None,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_size = int(hidden_size)
        self.output_dim = output_dim
        self.dtype = dtype

        self.x_embedder = nn.Linear(output_dim, self.hidden_size)
        self.t_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)
        self.freq_embedder = TimestepEmbedder(self.hidden_size, dtype=dtype)
        self.x_pos_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_size))

        self.blocks = nn.ModuleList([
            RDTBlock(self.hidden_size, num_heads) for _ in range(depth)
        ])
        self.out_proj = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.SiLU(),
            nn.Linear(512, output_dim),
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize position embeddings.
        x_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.hidden_size, torch.arange(1)
        )
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        # Initialize timestep and control freq embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.freq_embedder.mlp[2].weight, std=0.02)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def forward(
        self,
        x,
        freq,
        t,
        image_tokens,
        image_mask=None,
        text_tokens=None,
        text_mask=None,
    ):
        """
        Forward pass of RDT.
        
        x: (B, 1, 128), noisy target embedding token.
        freq: (B,), a scalar indicating control frequency.
        t: (B,) or (1,), diffusion timesteps.
        image_tokens: (B, L_img, 1024), non-language conditioning tokens.
        text_tokens: (B, L_txt, 1024), language conditioning tokens.
        """
        if x.ndim != 3 or x.shape[1] != self.horizon or x.shape[2] != self.output_dim:
            raise ValueError(
                f"Expected x with shape (B, {self.horizon}, {self.output_dim}), got {tuple(x.shape)}."
            )
        if image_tokens is None or image_tokens.ndim != 3:
            raise ValueError(
                "`image_tokens` must be provided with shape (B, image_context_len, hidden_size)."
            )
        if text_tokens is None or text_tokens.ndim != 3:
            raise ValueError(
                "`text_tokens` must be provided with shape (B, text_context_len, hidden_size)."
            )
        if image_tokens.shape[0] != x.shape[0] or text_tokens.shape[0] != x.shape[0]:
            raise ValueError(
                "Batch size mismatch among `x`, `image_tokens`, and `text_tokens`: "
                f"{x.shape[0]} vs {image_tokens.shape[0]} vs {text_tokens.shape[0]}."
            )

        x = x.to(device=self.x_embedder.weight.device, dtype=self.x_embedder.weight.dtype)
        x = self.x_embedder(x)

        t = self.t_embedder(t).unsqueeze(1)             # (B, 1, D) or (1, 1, D)
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1, -1)
        x = x + t + self.x_pos_embed

        if freq is not None:
            freq = self.freq_embedder(freq).unsqueeze(1)
            x = x + freq

        image_tokens = image_tokens.to(device=x.device, dtype=x.dtype)
        text_tokens = text_tokens.to(device=x.device, dtype=x.dtype)
        if image_mask is not None:
            image_mask = image_mask.to(device=x.device, dtype=torch.bool)
        if text_mask is not None:
            text_mask = text_mask.to(device=x.device, dtype=torch.bool)

        # Forward pass
        for block in self.blocks:
            x = block(
                x,
                image_tokens=image_tokens,
                image_mask=image_mask,
                text_tokens=text_tokens,
                text_mask=text_mask,
            )

        x = x[:, 0:1, :]
        x = self.out_proj(x)
        return x
