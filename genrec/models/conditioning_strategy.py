from __future__ import annotations

from typing import Any


def resolve_hybrid_conditioning_strategy(conditioning_cfg: dict[str, Any]) -> dict[str, Any]:
    layer_cfg = conditioning_cfg.get("layer_injection", {})
    training_dropout_cfg = conditioning_cfg.get("training_dropout", {})
    branch_dropout_cfg = training_dropout_cfg.get("branch_dropout", {})
    history_token_keep_cfg = training_dropout_cfg.get("history_token_keep_rate", {})

    return {
        "text_injection_mode": str(layer_cfg.get("text", "all")),
        "image_injection_mode": str(layer_cfg.get("image", "all")),
        "cf_injection_mode": str(layer_cfg.get("cf", "all")),
        "popularity_injection_mode": str(layer_cfg.get("popularity", "all")),
        "text_branch_dropout": float(branch_dropout_cfg.get("text", 0.0)),
        "image_branch_dropout": float(branch_dropout_cfg.get("image", 0.0)),
        "cf_branch_dropout": float(branch_dropout_cfg.get("cf", 0.0)),
        "popularity_branch_dropout": float(branch_dropout_cfg.get("popularity", 0.0)),
        "text_history_token_keep_rate": float(history_token_keep_cfg.get("text", 1.0)),
        "image_history_token_keep_rate": float(history_token_keep_cfg.get("image", 1.0)),
        "cf_history_token_keep_rate": float(history_token_keep_cfg.get("cf", 1.0)),
        "popularity_history_token_keep_rate": float(history_token_keep_cfg.get("popularity", 1.0)),
        "keep_pooled_condition_tokens": bool(
            training_dropout_cfg.get("keep_pooled_condition_tokens", True)
        ),
    }
