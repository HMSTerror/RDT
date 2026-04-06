import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyImageProcessor:
    def __init__(self, image_size=224):
        self.size = {"height": image_size, "width": image_size}
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]

    def preprocess(self, image, return_tensors="pt"):
        image = image.resize((self.size["width"], self.size["height"]))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - np.array(self.image_mean, dtype=np.float32)) / np.array(
            self.image_std, dtype=np.float32
        )
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        if return_tensors == "pt":
            return {"pixel_values": pixel_values.unsqueeze(0)}
        return {"pixel_values": pixel_values}


class DummyVisionTower(nn.Module):
    """
    Lightweight offline-only vision encoder for smoke tests.
    """

    def __init__(self, hidden_size=1152, image_size=224, patch_size=16):
        super().__init__()
        self._hidden_size = hidden_size
        self._image_size = image_size
        self._patch_size = patch_size
        self.image_processor = DummyImageProcessor(image_size=image_size)
        patch_dim = 3 * patch_size * patch_size
        self.patch_proj = nn.Linear(patch_dim, hidden_size)

    @torch.no_grad()
    def forward(self, images):
        if images.ndim != 4:
            raise ValueError(f"Expected images with shape (B,C,H,W), got {images.shape}")

        if images.shape[-1] != self._image_size or images.shape[-2] != self._image_size:
            images = F.interpolate(
                images, size=(self._image_size, self._image_size), mode="bilinear", align_corners=False
            )

        bsz, ch, h, w = images.shape
        ps = self._patch_size
        h_grid = h // ps
        w_grid = w // ps
        patches = images.unfold(2, ps, ps).unfold(3, ps, ps)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(bsz, h_grid * w_grid, ch * ps * ps)
        feats = self.patch_proj(patches)
        return feats.to(images.dtype)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches_per_side(self):
        return self._image_size // self._patch_size

    @property
    def num_patches(self):
        side = self.num_patches_per_side
        return side * side
