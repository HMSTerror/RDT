import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from models.hub_mixin import CompatiblePyTorchModelHubMixin
from models.rdt.model import RDT


class RDTRunner(
        nn.Module, 
        CompatiblePyTorchModelHubMixin, 
        repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"
    ):
    def __init__(self, *, action_dim, pred_horizon, config, 
                 lang_token_dim=None, img_token_dim=None, state_token_dim=None, 
                 max_lang_cond_len=None, img_cond_len=None, lang_pos_embed_config=None, 
                 img_pos_embed_config=None, dtype=torch.bfloat16,
                 num_items=None, item_latent_init=None,
                 diffusion_loss_weight=1.0, ranking_loss_weight=1.0,
                 ranking_temperature=0.07, item_latent_align_weight=0.0):
        super(RDTRunner, self).__init__()
        self.condition_encoder = None
        # Create diffusion model
        hidden_size = 1024
        if int(pred_horizon) != 1:
            raise ValueError(
                f"RecSys-DiT requires `pred_horizon == 1`, but got {pred_horizon}."
            )
        if int(action_dim) != 128:
            raise ValueError(
                f"RecSys-DiT requires `action_dim == 128`, but got {action_dim}."
            )
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            dtype=dtype,
        )
        
        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']

        self.pred_horizon = int(pred_horizon)
        self.action_dim = action_dim
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_temperature = float(ranking_temperature)
        self.item_latent_align_weight = float(item_latent_align_weight)
        self.num_items = int(num_items) if num_items is not None else 0
        self.item_latents = None
        self._init_item_latent_table(item_latent_init)

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.model.parameters()]))

    def _init_item_latent_table(self, item_latent_init):
        if item_latent_init is None and self.num_items <= 0:
            return

        if item_latent_init is not None:
            if not isinstance(item_latent_init, torch.Tensor):
                item_latent_init = torch.tensor(item_latent_init)
            item_latent_init = item_latent_init.detach().cpu().to(torch.float32)
            if item_latent_init.ndim != 2 or item_latent_init.shape[1] != self.action_dim:
                raise ValueError(
                    f"`item_latent_init` must have shape [num_items, {self.action_dim}], "
                    f"got {tuple(item_latent_init.shape)}."
                )
            inferred_num_items = int(item_latent_init.shape[0])
            if self.num_items and self.num_items != inferred_num_items:
                raise ValueError(
                    "Mismatch between `num_items` and `item_latent_init`: "
                    f"{self.num_items} vs {inferred_num_items}."
                )
            self.num_items = inferred_num_items
            self.item_latents = nn.Embedding(self.num_items, self.action_dim)
            with torch.no_grad():
                self.item_latents.weight.copy_(item_latent_init)
            return

        self.item_latents = nn.Embedding(self.num_items, self.action_dim)
        nn.init.normal_(self.item_latents.weight, std=0.02)

    def has_trainable_item_latents(self) -> bool:
        return self.item_latents is not None and self.num_items > 0

    def lookup_item_latents(
        self,
        item_ids: torch.Tensor,
        *,
        normalize: bool = False,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if not self.has_trainable_item_latents():
            raise RuntimeError("Trainable item latent table is not initialized.")

        item_ids = item_ids.to(device=self.item_latents.weight.device, dtype=torch.long)
        latents = self.item_latents(item_ids)
        if normalize:
            latents = F.normalize(latents.float(), dim=-1)
        if dtype is not None:
            latents = latents.to(dtype=dtype)
        return latents

    def get_item_latent_table(
        self,
        *,
        normalize: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor | None:
        if not self.has_trainable_item_latents():
            return None

        latents = self.item_latents.weight
        if normalize:
            latents = F.normalize(latents.float(), dim=-1)
        if device is not None or dtype is not None:
            latents = latents.to(device=device or latents.device, dtype=dtype or latents.dtype)
        return latents

    def set_condition_encoder(self, condition_encoder):
        self.condition_encoder = condition_encoder
        return self

    def encode_conditions(
        self,
        *,
        history_id_embeds,
        history_pixel_values,
        target_pixel_values,
        text,
    ):
        if self.condition_encoder is None:
            raise RuntimeError(
                "Condition encoding requires `self.condition_encoder` to be set. "
                "Attach a ConditionEncoder via `rdt_runner.set_condition_encoder(...)` first."
            )

        cond_device = self.condition_encoder.history_image_proj.weight.device
        history_id_embeds = history_id_embeds.to(device=cond_device)
        history_pixel_values = history_pixel_values.to(device=cond_device)
        target_pixel_values = target_pixel_values.to(device=cond_device)

        cond_outputs = self.condition_encoder(
            history_id_embeds=history_id_embeds,
            history_pixel_values=history_pixel_values,
            target_pixel_values=target_pixel_values,
            text=text,
        )
        model_device = self.model.x_embedder.weight.device
        model_dtype = self.model.x_embedder.weight.dtype
        return {
            "image_tokens": cond_outputs["image_tokens"].to(
                device=model_device,
                dtype=model_dtype,
            ),
            "image_attention_mask": cond_outputs["image_attention_mask"].to(
                device=model_device,
                dtype=torch.bool,
            ),
            "text_tokens": cond_outputs["text_tokens"].to(
                device=model_device,
                dtype=model_dtype,
            ),
            "text_attention_mask": cond_outputs["text_attention_mask"].to(
                device=model_device,
                dtype=torch.bool,
            ),
        }

    def _prepare_condition_dict(self, condition_dict):
        model_device = self.model.x_embedder.weight.device
        model_dtype = self.model.x_embedder.weight.dtype
        return {
            "image_tokens": condition_dict["image_tokens"].to(
                device=model_device,
                dtype=model_dtype,
            ),
            "image_attention_mask": condition_dict["image_attention_mask"].to(
                device=model_device,
                dtype=torch.bool,
            ) if condition_dict.get("image_attention_mask") is not None else None,
            "text_tokens": condition_dict["text_tokens"].to(
                device=model_device,
                dtype=model_dtype,
            ),
            "text_attention_mask": condition_dict["text_attention_mask"].to(
                device=model_device,
                dtype=torch.bool,
            ) if condition_dict.get("text_attention_mask") is not None else None,
        }

    # ========= Train  ============
    def compute_loss(
        self,
        *,
        action_gt=None,
        target_item_ids=None,
        ctrl_freqs,
        condition_dict=None,
        history_id_embeds=None,
        history_pixel_values=None,
        target_pixel_values=None,
        text=None,
        return_loss_dict: bool = False,
    ) -> torch.Tensor:
        '''
        action_gt: (batch_size, 1, state_token_dim), ground-truth target embeddings for supervision
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        if target_item_ids is not None and self.has_trainable_item_latents():
            target_latents = self.lookup_item_latents(
                target_item_ids,
                dtype=self.model.x_embedder.weight.dtype,
            ).unsqueeze(1)
            if action_gt is not None and (
                action_gt.ndim != 3 or action_gt.shape[1] != 1 or action_gt.shape[2] != self.action_dim
            ):
                raise ValueError(
                    f"Expected `action_gt` with shape (B, 1, {self.action_dim}), got {tuple(action_gt.shape)}."
                )
        elif action_gt is not None:
            if action_gt.ndim != 3 or action_gt.shape[1] != 1:
                raise ValueError(
                    f"Expected `action_gt` with shape (B, 1, {self.action_dim}), got {tuple(action_gt.shape)}."
                )
            target_latents = action_gt.to(dtype=self.model.x_embedder.weight.dtype)
        else:
            raise ValueError(
                "`compute_loss` requires either `target_item_ids` with a trainable item latent table "
                "or `action_gt`."
            )

        batch_size = target_latents.shape[0]
        device = target_latents.device  

        # Sample noise that we'll add to the actions
        noise = torch.randn(
            target_latents.shape, dtype=target_latents.dtype, device=device
        )
        # Sample random diffusion timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, 
            (batch_size,), device=device
        ).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(
            target_latents, noise, timesteps)

        if condition_dict is None:
            if any(value is None for value in (
                history_id_embeds,
                history_pixel_values,
                target_pixel_values,
                text,
            )):
                raise ValueError(
                    "`compute_loss` requires either `condition_dict` or "
                    "the raw multimodal inputs (`history_id_embeds`, `history_pixel_values`, "
                    "`target_pixel_values`, `text`)."
                )
            condition_dict = self.encode_conditions(
                history_id_embeds=history_id_embeds,
                history_pixel_values=history_pixel_values,
                target_pixel_values=target_pixel_values,
                text=text,
            )
        else:
            condition_dict = self._prepare_condition_dict(condition_dict)

        pred_x0 = self.model(
            noisy_action,
            ctrl_freqs,
            timesteps,
            image_tokens=condition_dict["image_tokens"],
            image_mask=condition_dict["image_attention_mask"],
            text_tokens=condition_dict["text_tokens"],
            text_mask=condition_dict["text_attention_mask"],
        )

        diffusion_loss = F.mse_loss(pred_x0.float(), target_latents.float())
        total_loss = self.diffusion_loss_weight * diffusion_loss

        ranking_loss = None
        if (
            self.ranking_loss_weight > 0
            and target_item_ids is not None
            and self.has_trainable_item_latents()
        ):
            pred_query = F.normalize(pred_x0[:, 0, :].float(), dim=-1)
            item_table = self.get_item_latent_table(normalize=True, device=pred_query.device)
            logits = pred_query @ item_table.t()
            logits = logits / max(self.ranking_temperature, 1e-6)
            ranking_loss = F.cross_entropy(
                logits,
                target_item_ids.to(device=pred_query.device, dtype=torch.long),
            )
            total_loss = total_loss + self.ranking_loss_weight * ranking_loss

        align_loss = None
        if (
            self.item_latent_align_weight > 0
            and action_gt is not None
            and target_item_ids is not None
            and self.has_trainable_item_latents()
        ):
            target_static = F.normalize(action_gt[:, 0, :].float(), dim=-1)
            target_lookup = self.lookup_item_latents(
                target_item_ids,
                normalize=True,
            ).float()
            align_loss = F.mse_loss(target_lookup, target_static)
            total_loss = total_loss + self.item_latent_align_weight * align_loss

        if return_loss_dict:
            loss_dict = {
                "loss": total_loss.detach(),
                "diffusion_loss": diffusion_loss.detach(),
            }
            if ranking_loss is not None:
                loss_dict["ranking_loss"] = ranking_loss.detach()
            if align_loss is not None:
                loss_dict["item_latent_align_loss"] = align_loss.detach()
            return total_loss, loss_dict

        return total_loss

    @torch.no_grad()
    def sample(self, history_id_embeds, history_pixel_values, text, target_pixel_values=None):
        '''
        history_id_embeds: (B, history_len, 128)
        history_pixel_values: (B, history_len, 3, 224, 224)
        text: list[str] of length B
        target_pixel_values: optional (B, 3, 224, 224). If omitted, uses zeros.

        return: (B, 1, 128), denoised target embedding
        '''
        if self.condition_encoder is None:
            raise RuntimeError(
                "`sample()` requires `self.condition_encoder` to be set. "
                "Attach a ConditionEncoder via `rdt_runner.set_condition_encoder(...)` first."
            )

        if history_id_embeds.ndim != 3:
            raise ValueError(
                f"`history_id_embeds` must have shape [B, history_len, 128], got {tuple(history_id_embeds.shape)}."
            )
        if history_pixel_values.ndim != 5:
            raise ValueError(
                "`history_pixel_values` must have shape [B, history_len, 3, 224, 224], "
                f"got {tuple(history_pixel_values.shape)}."
            )

        batch_size = history_id_embeds.shape[0]
        if history_pixel_values.shape[0] != batch_size or len(text) != batch_size:
            raise ValueError("Batch size mismatch among history_id_embeds, history_pixel_values, and text.")
        if target_pixel_values is not None:
            if target_pixel_values.ndim != 4:
                raise ValueError(
                    "`target_pixel_values` must have shape [B, 3, 224, 224], "
                    f"got {tuple(target_pixel_values.shape)}."
                )
            if target_pixel_values.shape[0] != batch_size:
                raise ValueError(
                    "Batch size mismatch among history_id_embeds, history_pixel_values, "
                    "target_pixel_values, and text."
                )

        device = self.model.x_embedder.weight.device
        model_dtype = self.model.x_embedder.weight.dtype
        cond_device = history_pixel_values.device
        if target_pixel_values is None:
            target_pixel_values = torch.zeros(
                batch_size, 3, 224, 224,
                device=cond_device,
                dtype=history_pixel_values.dtype,
            )
        else:
            target_pixel_values = target_pixel_values.to(
                device=cond_device,
                dtype=history_pixel_values.dtype,
            )
        dummy_target_pixel_values = torch.zeros_like(target_pixel_values)

        self.model.eval()
        self.condition_encoder.eval()

        condition_cond = self.encode_conditions(
            history_id_embeds=history_id_embeds,
            history_pixel_values=history_pixel_values,
            target_pixel_values=target_pixel_values,
            text=text,
        )

        condition_uncond = self.encode_conditions(
            history_id_embeds=torch.zeros_like(history_id_embeds),
            history_pixel_values=torch.zeros_like(history_pixel_values),
            target_pixel_values=dummy_target_pixel_values,
            text=[""] * batch_size,
        )

        x_t = torch.randn(batch_size, 1, self.action_dim, device=device, dtype=model_dtype)
        ctrl_freqs = torch.ones(batch_size, device=device, dtype=torch.float32)
        guidance_scale = 3.0

        ddim_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule=self.noise_scheduler.config.beta_schedule,
            prediction_type="sample",
            clip_sample=False,
        )
        ddim_scheduler.set_timesteps(50)

        for t in ddim_scheduler.timesteps:
            timestep = torch.full((batch_size,), int(t), device=device, dtype=torch.long)

            noise_pred_cond = self.model(
                x_t,
                ctrl_freqs,
                timestep,
                image_tokens=condition_cond["image_tokens"],
                image_mask=condition_cond["image_attention_mask"],
                text_tokens=condition_cond["text_tokens"],
                text_mask=condition_cond["text_attention_mask"],
            )
            noise_pred_uncond = self.model(
                x_t,
                ctrl_freqs,
                timestep,
                image_tokens=condition_uncond["image_tokens"],
                image_mask=condition_uncond["image_attention_mask"],
                text_tokens=condition_uncond["text_tokens"],
                text_mask=condition_uncond["text_attention_mask"],
            )
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            x_t = ddim_scheduler.step(noise_pred, t, x_t).prev_sample
            x_t = x_t.to(dtype=model_dtype)

        return x_t
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
