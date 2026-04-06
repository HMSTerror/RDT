import argparse

import torch

from models.multimodal_encoder.condition_encoder import ConditionEncoder
from models.multimodal_encoder.dummy_encoder import DummyVisionTower
from models.multimodal_encoder.dummy_text import DummyTextEncoder, DummyTokenizer
from models.rdt_runner import RDTRunner


def build_dummy_batch(batch_size, history_len, device, dtype):
    history_id_embeds = torch.randn(batch_size, history_len, 128, device=device, dtype=dtype)
    history_pixel_values = torch.randn(batch_size, history_len, 3, 224, 224, device=device, dtype=dtype)
    target_pixel_values = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
    target_embed = torch.randn(batch_size, 1, 128, device=device, dtype=dtype)
    text = [
        "user watched trail shoes and clicked hiking backpack",
        "user compared guitar strings and opened acoustic tuner",
    ]
    return {
        "history_id_embeds": history_id_embeds,
        "history_pixel_values": history_pixel_values,
        "target_pixel_values": target_pixel_values,
        "target_embed": target_embed,
        "text": text[:batch_size],
    }


def main():
    parser = argparse.ArgumentParser(description="Single-batch RecSys-DiT overfit smoke test.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--history-len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    dummy_vision = DummyVisionTower(hidden_size=32, image_size=224, patch_size=16)
    dummy_tokenizer = DummyTokenizer(vocab_size=512)
    dummy_text_encoder = DummyTextEncoder(vocab_size=512, hidden_size=48)

    condition_encoder = ConditionEncoder(
        history_len=args.history_len,
        hidden_dim=1024,
        vision_encoder=dummy_vision,
        text_tokenizer=dummy_tokenizer,
        text_encoder=dummy_text_encoder,
        max_text_length=64,
        device=device,
        torch_dtype=dtype,
    )

    config = {
        "rdt": {
            "depth": 2,
            "num_heads": 4,
        },
        "noise_scheduler": {
            "num_train_timesteps": 100,
            "num_inference_timesteps": 10,
            "beta_schedule": "linear",
            "prediction_type": "sample",
            "clip_sample": False,
        },
    }
    runner = RDTRunner(
        action_dim=128,
        pred_horizon=1,
        config=config,
        dtype=dtype,
    ).set_condition_encoder(condition_encoder)
    runner.to(device)
    runner.train()

    optimizer = torch.optim.AdamW(
        (param for param in runner.parameters() if param.requires_grad),
        lr=args.lr,
    )

    batch = build_dummy_batch(
        batch_size=args.batch_size,
        history_len=args.history_len,
        device=device,
        dtype=dtype,
    )
    ctrl_freqs = torch.ones(args.batch_size, device=device, dtype=torch.float32)

    initial_loss = None
    final_loss = None
    best_loss = None
    fixed_seed = 1234

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)

        torch.manual_seed(fixed_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fixed_seed)

        loss = runner(
            history_id_embeds=batch["history_id_embeds"],
            history_pixel_values=batch["history_pixel_values"],
            target_pixel_values=batch["target_pixel_values"],
            text=batch["text"],
            action_gt=batch["target_embed"],
            ctrl_freqs=ctrl_freqs,
        )
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if initial_loss is None:
            initial_loss = loss_value
        final_loss = loss_value
        best_loss = loss_value if best_loss is None else min(best_loss, loss_value)
        print(f"step {step:02d} | loss = {loss_value:.6f}")

    print("")
    print(f"initial_loss = {initial_loss:.6f}")
    print(f"best_loss    = {best_loss:.6f}")
    print(f"final_loss   = {final_loss:.6f}")
    if best_loss < initial_loss:
        print("Overfit smoke test passed: loss decreased on the fixed dummy batch.")
    else:
        print("Overfit smoke test warning: loss did not decrease. Inspect gradients and learning rate.")


if __name__ == "__main__":
    main()
