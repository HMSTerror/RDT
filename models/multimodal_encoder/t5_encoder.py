from pathlib import Path

import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5Embedder:
    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        allow_device_map=False,
        model_max_length=120,
        local_files_only=False,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir
        self.allow_device_map = bool(allow_device_map)

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }
        else:
            t5_model_kwargs = dict(t5_model_kwargs)

        if use_offload_folder is not None:
            if not self.allow_device_map:
                raise ValueError(
                    "`use_offload_folder` requires `allow_device_map=True`. "
                    "This path is intended for single-process large-model offload only."
                )
            t5_model_kwargs["offload_folder"] = use_offload_folder
            t5_model_kwargs["device_map"] = {
                "shared": self.device,
                "encoder.embed_tokens": self.device,
                "encoder.block.0": self.device,
                "encoder.block.1": self.device,
                "encoder.block.2": self.device,
                "encoder.block.3": self.device,
                "encoder.block.4": self.device,
                "encoder.block.5": self.device,
                "encoder.block.6": self.device,
                "encoder.block.7": self.device,
                "encoder.block.8": self.device,
                "encoder.block.9": self.device,
                "encoder.block.10": self.device,
                "encoder.block.11": self.device,
                "encoder.block.12": "disk",
                "encoder.block.13": "disk",
                "encoder.block.14": "disk",
                "encoder.block.15": "disk",
                "encoder.block.16": "disk",
                "encoder.block.17": "disk",
                "encoder.block.18": "disk",
                "encoder.block.19": "disk",
                "encoder.block.20": "disk",
                "encoder.block.21": "disk",
                "encoder.block.22": "disk",
                "encoder.block.23": "disk",
                "encoder.final_layer_norm": "disk",
                "encoder.dropout": "disk",
            }
        elif self.allow_device_map:
            # Single-process opt-in path for explicit HF placement without disk offload.
            t5_model_kwargs["device_map"] = {
                "shared": self.device,
                "encoder": self.device,
            }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        if from_pretrained is None:
            raise ValueError("`from_pretrained` must be a Hugging Face model id or a local T5 checkpoint directory.")

        pretrained_path = Path(str(from_pretrained))
        is_local_checkpoint = pretrained_path.exists()
        model_id = str(from_pretrained)
        if (not is_local_checkpoint) and ("t5" not in model_id.lower()):
            raise ValueError(
                "Unsupported text encoder source. Expected a T5-family Hugging Face model id "
                f"or a local T5 checkpoint directory, but got `{from_pretrained}`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **t5_model_kwargs,
        ).eval()
        if getattr(self.model, "hf_device_map", None) is None:
            self.model.to(self.device)
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask
