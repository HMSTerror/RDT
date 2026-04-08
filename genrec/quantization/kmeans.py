from __future__ import annotations

import numpy as np


def squared_l2_distance(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    x_sq = np.sum(x * x, axis=1, keepdims=True)
    c_sq = np.sum(centroids * centroids, axis=1, keepdims=True).T
    distances = x_sq + c_sq - 2.0 * (x @ centroids.T)
    # Numerical safety: finite + non-negative squared distances.
    distances = np.nan_to_num(distances, nan=np.inf, posinf=np.inf, neginf=np.inf)
    return np.maximum(distances, 0.0, out=distances)


def assign_clusters(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = squared_l2_distance(x, centroids)
    return np.argmin(distances, axis=1).astype(np.int32)


def kmeans_plus_plus_init(x: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] < n_clusters:
        raise ValueError(
            f"Cannot initialize {n_clusters} centroids from only {x.shape[0]} samples."
        )

    centroids = np.empty((n_clusters, x.shape[1]), dtype=np.float32)
    first_idx = int(rng.integers(0, x.shape[0]))
    centroids[0] = x[first_idx]

    closest_dist_sq = squared_l2_distance(x, centroids[:1]).reshape(-1).astype(np.float64, copy=False)
    for i in range(1, n_clusters):
        closest_dist_sq = np.nan_to_num(closest_dist_sq, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(closest_dist_sq, 0.0, out=closest_dist_sq)

        total = float(np.sum(closest_dist_sq, dtype=np.float64))
        if not np.isfinite(total) or total <= 1e-12:
            # Degenerate case (e.g., all points identical after sanitization).
            probs = None
        else:
            probs = closest_dist_sq / total
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(probs, 0.0, out=probs)
            prob_sum = float(np.sum(probs, dtype=np.float64))
            probs = probs / prob_sum if prob_sum > 1e-12 else None

        if probs is None:
            next_idx = int(rng.integers(0, x.shape[0]))
        else:
            next_idx = int(rng.choice(x.shape[0], p=probs))
        centroids[i] = x[next_idx]
        new_dist_sq = squared_l2_distance(x, centroids[i : i + 1]).reshape(-1).astype(np.float64, copy=False)
        new_dist_sq = np.nan_to_num(new_dist_sq, nan=np.inf, posinf=np.inf, neginf=np.inf)
        np.maximum(new_dist_sq, 0.0, out=new_dist_sq)
        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

    return centroids


def fit_kmeans(
    x: np.ndarray,
    n_clusters: int,
    *,
    n_iters: int = 25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(x.shape)}.")
    if x.shape[0] < n_clusters:
        raise ValueError(
            f"Need at least {n_clusters} rows for k-means, got {x.shape[0]}."
        )

    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus_init(x, n_clusters=n_clusters, rng=rng)

    labels = np.zeros(x.shape[0], dtype=np.int32)
    for _ in range(max(1, int(n_iters))):
        labels = assign_clusters(x, centroids)
        for cluster_idx in range(n_clusters):
            mask = labels == cluster_idx
            if not np.any(mask):
                replacement_idx = int(rng.integers(0, x.shape[0]))
                centroids[cluster_idx] = x[replacement_idx]
                continue
            centroids[cluster_idx] = x[mask].mean(axis=0)

    distances = squared_l2_distance(x, centroids)
    inertia = float(np.take_along_axis(distances, labels[:, None], axis=1).sum())
    return centroids.astype(np.float32), labels.astype(np.int32), inertia
