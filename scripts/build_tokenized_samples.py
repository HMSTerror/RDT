#!/usr/bin/env python
# coding=utf-8

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from genrec.tokenization.semantic_ids import (  # noqa: E402
    build_layout_from_item_to_code,
    load_item_to_code,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build semantic-ID tokenized train/val/test samples from split buffers."
    )
    parser.add_argument("--buffer-root", type=Path, required=True)
    parser.add_argument("--semantic-id-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--pad-token-id", type=int, default=0)
    parser.add_argument("--mask-token-id", type=int, default=1)
    parser.add_argument("--cls-token-id", type=int, default=2)
    parser.add_argument("--sep-token-id", type=int, default=3)
    parser.add_argument("--eos-token-id", type=int, default=4)
    return parser.parse_args()


def resolve_split_root(buffer_root: Path, split_name: str) -> Path:
    candidate = buffer_root / split_name
    if (candidate / "stats.json").exists():
        return candidate
    if (buffer_root / "stats.json").exists():
        return buffer_root
    raise FileNotFoundError(f"Could not resolve split `{split_name}` under {buffer_root}.")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def collect_split_arrays(split_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    chunk_dirs = sorted(
        [
            path
            for path in split_root.glob("chunk_*")
            if path.is_dir() and (path / "samples.npz").exists()
        ],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not chunk_dirs:
        raise RuntimeError(f"No chunk directories found under {split_root}.")

    history_ids_list: List[np.ndarray] = []
    history_mask_list: List[np.ndarray] = []
    target_ids_list: List[np.ndarray] = []

    for chunk_dir in tqdm(chunk_dirs, desc=f"Loading chunks from {split_root.name}", unit="chunk"):
        with np.load(chunk_dir / "samples.npz", allow_pickle=False) as payload:
            history_ids_list.append(payload["history_item_ids"].copy())
            history_mask_list.append(payload["history_mask"].copy())
            target_ids_list.append(payload["target_item_ids"].copy())

    history_item_ids = np.concatenate(history_ids_list, axis=0).astype(np.int32, copy=False)
    history_mask = np.concatenate(history_mask_list, axis=0).astype(np.bool_, copy=False)
    target_item_ids = np.concatenate(target_ids_list, axis=0).astype(np.int32, copy=False)
    return history_item_ids, history_mask, target_item_ids


def build_sample_arrays(
    *,
    history_item_ids: np.ndarray,
    history_mask: np.ndarray,
    target_item_ids: np.ndarray,
    idx_to_item: Dict[int, str],
    item_to_code: Dict[str, List[int]],
    layout,
) -> Dict[str, np.ndarray]:
    num_samples, history_len = history_item_ids.shape
    code_len = layout.code_len
    max_seq_len = 1 + history_len * (code_len + 1) + code_len + 1

    input_ids = np.full((num_samples, max_seq_len), layout.pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((num_samples, max_seq_len), dtype=np.bool_)
    labels = np.full((num_samples, max_seq_len), -100, dtype=np.int32)
    token_slot_ids = np.full((num_samples, max_seq_len), -1, dtype=np.int32)
    token_codebook_ids = np.full((num_samples, max_seq_len), -1, dtype=np.int32)
    seq_lengths = np.zeros((num_samples,), dtype=np.int32)
    history_semantic_ids = np.full((num_samples, history_len, code_len), -1, dtype=np.int32)
    target_semantic_ids = np.full((num_samples, code_len), -1, dtype=np.int32)

    for sample_idx in tqdm(range(num_samples), desc="Tokenizing semantic-ID samples", unit="sample"):
        pos = 0
        input_ids[sample_idx, pos] = layout.cls_token_id
        attention_mask[sample_idx, pos] = True
        pos += 1

        for history_slot in range(history_len):
            if not history_mask[sample_idx, history_slot]:
                continue
            item_idx = int(history_item_ids[sample_idx, history_slot])
            if item_idx < 0:
                continue
            item_id = idx_to_item[item_idx]
            codes = item_to_code[item_id]
            history_semantic_ids[sample_idx, history_slot] = np.asarray(codes, dtype=np.int32)

            encoded = layout.encode_codes(codes)
            for codebook_idx, token_id in enumerate(encoded):
                input_ids[sample_idx, pos] = int(token_id)
                attention_mask[sample_idx, pos] = True
                token_slot_ids[sample_idx, pos] = int(history_slot)
                token_codebook_ids[sample_idx, pos] = int(codebook_idx)
                pos += 1

            input_ids[sample_idx, pos] = layout.sep_token_id
            attention_mask[sample_idx, pos] = True
            token_slot_ids[sample_idx, pos] = int(history_slot)
            pos += 1

        target_item_idx = int(target_item_ids[sample_idx])
        target_item_id = idx_to_item[target_item_idx]
        target_codes = item_to_code[target_item_id]
        target_semantic_ids[sample_idx] = np.asarray(target_codes, dtype=np.int32)
        target_tokens = layout.encode_codes(target_codes)
        for codebook_idx, target_token_id in enumerate(target_tokens):
            input_ids[sample_idx, pos] = layout.mask_token_id
            attention_mask[sample_idx, pos] = True
            labels[sample_idx, pos] = int(target_token_id)
            token_slot_ids[sample_idx, pos] = int(history_len)
            token_codebook_ids[sample_idx, pos] = int(codebook_idx)
            pos += 1

        input_ids[sample_idx, pos] = layout.eos_token_id
        attention_mask[sample_idx, pos] = True
        pos += 1
        seq_lengths[sample_idx] = int(pos)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "token_slot_ids": token_slot_ids,
        "token_codebook_ids": token_codebook_ids,
        "seq_lengths": seq_lengths,
        "history_item_ids": history_item_ids.astype(np.int32, copy=False),
        "history_mask": history_mask.astype(np.bool_, copy=False),
        "target_item_ids": target_item_ids.astype(np.int32, copy=False),
        "history_semantic_ids": history_semantic_ids,
        "target_semantic_ids": target_semantic_ids,
    }


def main() -> None:
    args = parse_args()
    split_names = [split.strip() for split in args.splits.split(",") if split.strip()]

    item_to_code = load_item_to_code(args.semantic_id_root / "item_to_code.json")
    layout = build_layout_from_item_to_code(
        item_to_code,
        pad_token_id=args.pad_token_id,
        mask_token_id=args.mask_token_id,
        cls_token_id=args.cls_token_id,
        sep_token_id=args.sep_token_id,
        eos_token_id=args.eos_token_id,
    )

    args.output_root.mkdir(parents=True, exist_ok=True)

    split_summary = {}
    for split_name in split_names:
        split_root = resolve_split_root(args.buffer_root, split_name)
        item_map = load_json(split_root / "item_map.json")
        idx_to_item = {int(idx): item_id for item_id, idx in item_map.items()}
        history_item_ids, history_mask, target_item_ids = collect_split_arrays(split_root)
        arrays = build_sample_arrays(
            history_item_ids=history_item_ids,
            history_mask=history_mask,
            target_item_ids=target_item_ids,
            idx_to_item=idx_to_item,
            item_to_code=item_to_code,
            layout=layout,
        )

        split_output_dir = args.output_root / split_name
        split_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = split_output_dir / "samples.npz"
        np.savez_compressed(output_path, **arrays)
        split_summary[split_name] = {
            "num_samples": int(target_item_ids.shape[0]),
            "history_len": int(history_item_ids.shape[1]),
            "code_len": int(layout.code_len),
            "max_seq_len": int(arrays["input_ids"].shape[1]),
            "path": str(output_path),
        }

    manifest = {
        "buffer_root": str(args.buffer_root),
        "semantic_id_root": str(args.semantic_id_root),
        "output_root": str(args.output_root),
        "code_len": int(layout.code_len),
        "vocab_sizes": [int(value) for value in layout.vocab_sizes],
        "semantic_offsets": [int(value) for value in layout.semantic_offsets],
        "vocab_size": int(layout.vocab_size),
        "special_tokens": {
            "pad_token_id": int(layout.pad_token_id),
            "mask_token_id": int(layout.mask_token_id),
            "cls_token_id": int(layout.cls_token_id),
            "sep_token_id": int(layout.sep_token_id),
            "eos_token_id": int(layout.eos_token_id),
        },
        "splits": split_summary,
    }
    with open(args.output_root / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)

    print("========== Tokenized Semantic-ID Samples Built ==========")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
