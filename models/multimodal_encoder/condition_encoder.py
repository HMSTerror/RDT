from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder


class ConditionEncoder(nn.Module):
    """Fuse recommendation conditions into a shared hidden space."""

    def __init__(
        self,
        *,
        history_len: int,
        hidden_dim: int = 1024,
        vision_tower_name: str = "google/siglip-base-patch16-224",
        text_encoder_name: str = "google/t5-v1_1-xxl",
        max_text_length: int = 64,
        device: Optional[torch.device] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
        vision_encoder: Optional[nn.Module] = None,
        text_tokenizer=None,
        text_encoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.history_len = int(history_len)
        self.hidden_dim = int(hidden_dim)
        self.max_text_length = int(max_text_length)
        self._backbone_device = torch.device(device) if device is not None else torch.device("cpu")

        self.vision_encoder = vision_encoder
        if self.vision_encoder is None:
            self.vision_encoder = SiglipVisionTower(
                vision_tower=vision_tower_name,
                args=SimpleNamespace(mm_vision_select_feature="patch"),
            )

        if text_tokenizer is None or text_encoder is None:
            text_embedder = T5Embedder(
                device=self._backbone_device,
                from_pretrained=text_encoder_name,
                torch_dtype=torch_dtype,
                model_max_length=max_text_length,
                local_files_only=local_files_only,
            )
            self.text_tokenizer = text_embedder.tokenizer
            self.text_encoder = text_embedder.model
        else:
            self.text_tokenizer = text_tokenizer
            self.text_encoder = text_encoder

        self._freeze_backbones()

        self.vision_patch_dim = int(self.vision_encoder.hidden_size)
        if hasattr(self.text_encoder.config, "d_model"):
            self.text_embed_dim = int(self.text_encoder.config.d_model)
        elif hasattr(self.text_encoder.config, "hidden_size"):
            self.text_embed_dim = int(self.text_encoder.config.hidden_size)
        else:
            raise ValueError("Text encoder config must define `d_model` or `hidden_size`.")
        self.target_num_patches = int(self.vision_encoder.num_patches)
        if self.target_num_patches != 196:
            raise ValueError(
                "ConditionEncoder expects SigLIP patch tokens with length 196 for 224x224 inputs, "
                f"but got {self.target_num_patches}."
            )

        self.history_image_proj = nn.Linear(self.vision_patch_dim, self.hidden_dim)
        self.history_id_proj = nn.Linear(128, self.hidden_dim)
        self.target_image_proj = nn.Linear(self.vision_patch_dim, self.hidden_dim)
        self.text_proj = nn.Linear(self.text_embed_dim, self.hidden_dim)

        self.image_context_len = 2 * self.history_len + self.target_num_patches
        self.image_position_embeddings = nn.Parameter(
            torch.zeros(1, self.image_context_len, self.hidden_dim)
        )
        self.text_position_embeddings = nn.Parameter(
            torch.zeros(1, self.max_text_length, self.hidden_dim)
        )
        nn.init.trunc_normal_(self.image_position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.text_position_embeddings, std=0.02)

    def _freeze_backbones(self) -> None:
        self.vision_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vision_encoder.eval()
        self.text_encoder.eval()
        if hasattr(self.vision_encoder, "vision_tower"):
            self.vision_encoder.vision_tower.requires_grad_(False)
            self.vision_encoder.vision_tower.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_backbones()
        return self

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        parameter = next(module.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(module.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    def _encode_history_images(self, history_pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, history_len, channels, height, width = history_pixel_values.shape
        flat_history = history_pixel_values.reshape(batch_size * history_len, channels, height, width)

        self.vision_encoder.eval()
        if hasattr(self.vision_encoder, "vision_tower"):
            self.vision_encoder.vision_tower.eval()
        with torch.no_grad():
            history_patch_tokens = self.vision_encoder(flat_history)

        history_pooled = history_patch_tokens.mean(dim=1).reshape(batch_size, history_len, -1)
        history_pooled = history_pooled.to(
            device=self.history_image_proj.weight.device,
            dtype=self.history_image_proj.weight.dtype,
        )
        return self.history_image_proj(history_pooled)

    def _encode_history_ids(self, history_id_embeds: torch.Tensor) -> torch.Tensor:
        history_id_embeds = history_id_embeds.to(
            device=self.history_id_proj.weight.device,
            dtype=self.history_id_proj.weight.dtype,
        )
        return self.history_id_proj(history_id_embeds)

    def _encode_target_image(self, target_pixel_values: torch.Tensor) -> torch.Tensor:
        self.vision_encoder.eval()
        if hasattr(self.vision_encoder, "vision_tower"):
            self.vision_encoder.vision_tower.eval()
        with torch.no_grad():
            target_patch_tokens = self.vision_encoder(target_pixel_values)

        target_patch_tokens = target_patch_tokens.to(
            device=self.target_image_proj.weight.device,
            dtype=self.target_image_proj.weight.dtype,
        )
        return self.target_image_proj(target_patch_tokens)

    def _encode_text(self, text: Sequence[str]) -> Dict[str, torch.Tensor]:
        text_batch = list(text)
        tokenized = self.text_tokenizer(
            text_batch,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        text_device = self._module_device(self.text_encoder)
        input_ids = tokenized["input_ids"].to(text_device)
        attention_mask = tokenized["attention_mask"].to(text_device)

        self.text_encoder.eval()
        with torch.no_grad():
            text_hidden = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"]

        text_hidden = text_hidden.to(
            device=self.text_proj.weight.device,
            dtype=self.text_proj.weight.dtype,
        )
        return {
            "tokens": self.text_proj(text_hidden),
            "attention_mask": attention_mask.to(device=self.text_proj.weight.device, dtype=torch.bool),
        }

    def forward(
        self,
        *,
        history_id_embeds: torch.Tensor,
        history_pixel_values: torch.Tensor,
        target_pixel_values: torch.Tensor,
        text: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        if history_id_embeds.ndim != 3:
            raise ValueError(
                f"`history_id_embeds` must have shape [B, history_len, 128], got {tuple(history_id_embeds.shape)}"
            )
        if history_pixel_values.ndim != 5:
            raise ValueError(
                "`history_pixel_values` must have shape [B, history_len, 3, 224, 224], "
                f"got {tuple(history_pixel_values.shape)}"
            )
        if target_pixel_values.ndim != 4:
            raise ValueError(
                f"`target_pixel_values` must have shape [B, 3, 224, 224], got {tuple(target_pixel_values.shape)}"
            )

        batch_size, history_len, history_embed_dim = history_id_embeds.shape
        if history_len != self.history_len:
            raise ValueError(
                f"Expected history length {self.history_len}, but received {history_len}."
            )
        if history_embed_dim != 128:
            raise ValueError(
                f"Expected history id embed dim 128, but received {history_embed_dim}."
            )
        if len(text) != batch_size:
            raise ValueError(f"Expected {batch_size} text entries, but received {len(text)}.")

        history_image_tokens = self._encode_history_images(history_pixel_values)
        history_id_tokens = self._encode_history_ids(history_id_embeds)
        target_image_tokens = self._encode_target_image(target_pixel_values)
        text_outputs = self._encode_text(text)
        text_tokens = text_outputs["tokens"]
        text_attention_mask = text_outputs["attention_mask"]

        image_tokens = torch.cat(
            [
                history_image_tokens,
                history_id_tokens,
                target_image_tokens,
            ],
            dim=1,
        )
        image_tokens = image_tokens + self.image_position_embeddings[:, : image_tokens.shape[1]]
        text_tokens = text_tokens + self.text_position_embeddings[:, : text_tokens.shape[1]]

        image_branch_present = (
            history_id_embeds.detach().reshape(batch_size, -1).abs().sum(dim=1) > 0
        ) | (
            history_pixel_values.detach().reshape(batch_size, -1).abs().sum(dim=1) > 0
        ) | (
            target_pixel_values.detach().reshape(batch_size, -1).abs().sum(dim=1) > 0
        )

        image_branch_present = image_branch_present.to(device=image_tokens.device, dtype=torch.bool)
        image_tokens = image_tokens * image_branch_present[:, None, None].to(dtype=image_tokens.dtype)
        image_attention_mask = image_branch_present[:, None].expand(-1, image_tokens.shape[1])

        text_branch_present = torch.tensor(
            [bool(str(item).strip()) for item in text],
            device=text_tokens.device,
            dtype=torch.bool,
        )
        text_tokens = text_tokens * text_branch_present[:, None, None].to(dtype=text_tokens.dtype)
        text_attention_mask = text_attention_mask & text_branch_present[:, None]

        return {
            "image_tokens": image_tokens,
            "image_attention_mask": image_attention_mask,
            "text_tokens": text_tokens,
            "text_attention_mask": text_attention_mask,
        }
