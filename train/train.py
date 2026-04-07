#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import json
import logging
import math
import os
from types import SimpleNamespace
from pathlib import Path

import diffusers
import torch
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm

from models.multimodal_encoder.clip_encoder import CLIPVisionTower
from models.multimodal_encoder.condition_encoder import ConditionEncoder
from models.multimodal_encoder.dummy_encoder import DummyVisionTower
from models.multimodal_encoder.dummy_text import DummyTextEncoder, DummyTokenizer
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res


if is_wandb_available():
    import wandb


def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: mit
base_model: {base_model}
language:
- en
pipeline_tag: feature-extraction
library_name: transformers
tags:
- recommendation
- pytorch
- multimodal
- diffusion
- recsys
---
    """
    model_card = f"""
# RecSys-DiT - {repo_id}

This is a RecSys-DiT model derived from {base_model}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def train(args, logger):
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    if args.buffer_root is not None:
        config.setdefault("dataset", {})["buffer_root"] = args.buffer_root
    if args.image_root is not None:
        config.setdefault("dataset", {})["image_root"] = args.image_root

    buffer_root = (
        config.get("dataset", {}).get("buffer_root")
        or config.get("dataset", {}).get("preprocessed_buffer_root")
    )
    if buffer_root:
        buffer_root = Path(buffer_root)
        if not buffer_root.is_absolute():
            buffer_root = Path.cwd() / buffer_root
        stats_path = buffer_root / "stats.json"
        if stats_path.exists():
            with open(stats_path, "r", encoding="utf-8") as fp:
                buffer_stats = json.load(fp)
            buffer_history_len = int(buffer_stats["history_len"])
            original_dataset_history_len = config["dataset"].get("history_len")
            original_img_history_size = config["common"].get("img_history_size")
            if (
                original_dataset_history_len not in (None, buffer_history_len)
                or original_img_history_size != buffer_history_len
            ):
                logger.warning(
                    "Overriding history length from preprocessed buffer stats: "
                    f"dataset.history_len {original_dataset_history_len} -> {buffer_history_len}, "
                    f"common.img_history_size {original_img_history_size} -> {buffer_history_len}."
                )
            config["dataset"]["history_len"] = buffer_history_len
            config["common"]["img_history_size"] = buffer_history_len

    original_state_dim = config["common"].get("state_dim")
    original_action_chunk_size = config["common"].get("action_chunk_size")
    original_hidden_size = config["model"]["rdt"].get("hidden_size")
    if (
        original_state_dim != 128
        or original_action_chunk_size != 1
        or original_hidden_size != 1024
    ):
        logger.warning(
            "Overriding legacy config for RecSys-DiT: "
            f"state_dim {original_state_dim} -> 128, "
            f"action_chunk_size {original_action_chunk_size} -> 1, "
            f"rdt.hidden_size {original_hidden_size} -> 1024."
        )
    config["common"]["state_dim"] = 128
    config["common"]["action_chunk_size"] = 1
    config["model"]["state_token_dim"] = 128
    config["model"]["rdt"]["hidden_size"] = 1024

    logging_dir = Path(args.output_dir, args.logging_dir)
    report_to = None if args.report_to == "none" else args.report_to

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # The frozen multimodal backbones can run in reduced precision during training.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.pretrained_text_encoder_name_or_path is None:
        raise ValueError("`--pretrained_text_encoder_name_or_path` is required.")
    if args.pretrained_text_encoder_name_or_path.lower() == "dummy":
        tokenizer = DummyTokenizer(vocab_size=512)
        text_encoder = DummyTextEncoder(vocab_size=512, hidden_size=48)
    else:
        text_use_offload_folder = os.environ.get("RDT_T5_OFFLOAD_DIR")
        text_allow_device_map = False
        if accelerator.num_processes > 1:
            if text_use_offload_folder:
                logger.warning(
                    "Ignoring `RDT_T5_OFFLOAD_DIR` because distributed training requires "
                    "loading the frozen T5 encoder without `device_map`."
                )
            text_use_offload_folder = None
        elif text_use_offload_folder:
            text_allow_device_map = True

        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
            torch_dtype=weight_dtype,
            use_offload_folder=text_use_offload_folder,
            allow_device_map=text_allow_device_map,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_tower_name = args.pretrained_vision_encoder_name_or_path
    if vision_tower_name is None:
        raise ValueError("`--pretrained_vision_encoder_name_or_path` is required.")

    if ("clip" in vision_tower_name.lower()) and ("siglip" not in vision_tower_name.lower()):
        # CLIP backend for offline/local fallback if SigLIP is unavailable.
        clip_args = SimpleNamespace(mm_vision_select_layer=-2, mm_vision_select_feature="patch")
        vision_encoder = CLIPVisionTower(vision_tower=vision_tower_name, args=clip_args)
    elif vision_tower_name.lower() == "dummy":
        vision_encoder = DummyVisionTower(hidden_size=config["model"]["img_token_dim"])
    else:
        vision_encoder = SiglipVisionTower(vision_tower=vision_tower_name, args=None)
    image_processor = vision_encoder.image_processor

    # Load from a pretrained checkpoint
    if (
        args.pretrained_model_name_or_path is not None
        and not os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Constructing model from pretrained checkpoint.")
        rdt = RDTRunner.from_pretrained(args.pretrained_model_name_or_path)
    else:
        logger.info("Constructing model from provided config.")
        rdt = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            dtype=weight_dtype,
        )

    history_len = max(1, int(config["dataset"].get("history_len", config["common"]["img_history_size"])))
    condition_encoder = ConditionEncoder(
        history_len=history_len,
        hidden_dim=1024,
        vision_encoder=vision_encoder,
        text_tokenizer=tokenizer,
        text_encoder=text_encoder,
        text_encoder_name=args.pretrained_text_encoder_name_or_path,
        max_text_length=64,
        device=accelerator.device,
        torch_dtype=weight_dtype,
    )
    rdt.set_condition_encoder(condition_encoder)

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # which ensure saving model in huggingface format (config.json + pytorch_model.bin)
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model  # type: ignore
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)
    
    if args.gradient_checkpointing:
        # TODO: 
        raise NotImplementedError("Gradient checkpointing is not yet implemented.")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = [param for param in rdt.parameters() if param.requires_grad]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Dataset and DataLoaders creation:                                                           
    train_dataset = VLAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        img_history_size=config["common"]["img_history_size"],
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
    )
    sample_dataset = VLAConsumerDataset(
        config=config["dataset"],
        image_processor=image_processor,
        img_history_size=config["common"]["img_history_size"],
        image_aug=False,
        cond_mask_prob=0,
    )                              
    
    data_collator = DataCollatorForVLAConsumerDataset()                                                        
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=(args.dataloader_num_workers > 0)
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=(args.dataloader_num_workers > 0)
    )
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler                   
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process and report_to is not None:
        accelerator.init_trackers("recsysDiffusionTransformer", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Load from a pretrained checkpoint
    if (
        args.resume_from_checkpoint is None 
        and args.pretrained_model_name_or_path is not None
        and os.path.isfile(args.pretrained_model_name_or_path)
    ):
        logger.info("Loading from a pretrained checkpoint.")
        checkpoint = torch.load(args.pretrained_model_name_or_path, map_location="cpu")
        state_dict = checkpoint["module"] if isinstance(checkpoint, dict) and "module" in checkpoint else checkpoint
        accelerator.unwrap_model(rdt).load_state_dict(state_dict)
   
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path)) # load_module_strict=False
            except:
                logger.info("Resuming training state failed. Attempting to only load from model checkpoint.")
                checkpoint = torch.load(
                    os.path.join(args.output_dir, path, "pytorch_model", "mp_rank_00_model_states.pt"),
                    map_location="cpu",
                )
                state_dict = checkpoint["module"] if isinstance(checkpoint, dict) and "module" in checkpoint else checkpoint
                accelerator.unwrap_model(rdt).load_state_dict(state_dict)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):

        rdt.train()
        
        # Set the progress_bar to correct position
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)
        
        # Forward and backward...
        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                history_id_embeds = batch["history_id_embeds"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                history_pixel_values = batch["history_pixel_values"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                target_pixel_values = batch["target_pixel_values"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                target_embeds = batch["target_embed"].to(
                    device=accelerator.device,
                    dtype=weight_dtype,
                )
                ctrl_freqs = batch["ctrl_freqs"].to(
                    device=accelerator.device,
                    dtype=torch.float32,
                )

                loss = rdt(
                    history_id_embeds=history_id_embeds,
                    history_pixel_values=history_pixel_values,
                    target_pixel_values=target_pixel_values,
                    text=batch["text"],
                    action_gt=target_embeds,
                    ctrl_freqs=ctrl_freqs,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = rdt.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    sample_loss_for_log = log_sample_res(
                        rdt,    # We do not use EMA currently
                        args,
                        accelerator,
                        weight_dtype,
                        sample_dataset.get_dataset_id2name(),
                        sample_dataloader,
                        logger,
                    )
                    logger.info(sample_loss_for_log)
                    if report_to is not None:
                        accelerator.log(sample_loss_for_log, step=global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if report_to is not None:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_ok = True
        try:
            accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        except Exception as e:
            save_ok = False
            logger.warning(f"save_pretrained failed with {type(e).__name__}: {e}")
            logger.warning("Skipping final save due runtime serialization incompatibility.")

        if save_ok:
            logger.info(f"Saved Model to {args.output_dir}")

        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                token=args.hub_token,
                allow_patterns=["pytorch_model.bin", "*.json", "*.md"],
                # ignore_patterns=["step_*", "epoch_*"],
            )
            
    accelerator.end_training()
