from __future__ import annotations

from typing import Mapping

import numpy as np


def _ensure_2d_float32(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(array.shape)}.")
    return array


def _validate_aligned(named_embeddings: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not named_embeddings:
        raise ValueError("At least one embedding source is required for fusion.")

    validated: dict[str, np.ndarray] = {}
    item_count = None
    for name, matrix in named_embeddings.items():
        array = _ensure_2d_float32(matrix)
        if item_count is None:
            item_count = array.shape[0]
        elif array.shape[0] != item_count:
            raise ValueError(
                f"Embedding source `{name}` has {array.shape[0]} rows, expected {item_count}."
            )
        validated[name] = array
    return validated


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def weighted_sum_fusion(
    named_embeddings: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
    normalize_each: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    validated = _validate_aligned(named_embeddings)
    source_names = list(validated.keys())

    if weights is None:
        weights = {name: 1.0 for name in source_names}

    total_weight = float(sum(float(weights.get(name, 0.0)) for name in source_names))
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive value.")

    fused = None
    for name in source_names:
        matrix = validated[name]
        if normalize_each:
            matrix = l2_normalize(matrix)
        contribution = float(weights.get(name, 0.0)) * matrix
        fused = contribution if fused is None else (fused + contribution)

    fused = fused / total_weight
    if normalize_output:
        fused = l2_normalize(fused)
    return fused.astype(np.float32, copy=False)


def concat_fusion(
    named_embeddings: Mapping[str, np.ndarray],
    *,
    normalize_each: bool = True,
    normalize_output: bool = False,
) -> np.ndarray:
    validated = _validate_aligned(named_embeddings)
    matrices = []
    for matrix in validated.values():
        if normalize_each:
            matrix = l2_normalize(matrix)
        matrices.append(matrix)
    fused = np.concatenate(matrices, axis=1)
    if normalize_output:
        fused = l2_normalize(fused)
    return fused.astype(np.float32, copy=False)


def mean_fusion(
    named_embeddings: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float] | None = None,
    normalize_each: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    _ = weights
    return weighted_sum_fusion(
        named_embeddings,
        weights=None,
        normalize_each=normalize_each,
        normalize_output=normalize_output,
    )


FUSION_REGISTRY = {
    "weighted_sum": weighted_sum_fusion,
    "concat": concat_fusion,
    "mean": mean_fusion,
}


def fuse_embedding_dict(
    named_embeddings: Mapping[str, np.ndarray],
    *,
    strategy: str = "weighted_sum",
    weights: Mapping[str, float] | None = None,
    normalize_each: bool = True,
    normalize_output: bool = True,
) -> np.ndarray:
    if strategy not in FUSION_REGISTRY:
        raise ValueError(
            f"Unsupported fusion strategy `{strategy}`. "
            f"Available: {sorted(FUSION_REGISTRY)}."
        )

    if strategy == "concat":
        return concat_fusion(
            named_embeddings,
            normalize_each=normalize_each,
            normalize_output=normalize_output,
        )

    return FUSION_REGISTRY[strategy](
        named_embeddings,
        weights=weights,
        normalize_each=normalize_each,
        normalize_output=normalize_output,
    )
