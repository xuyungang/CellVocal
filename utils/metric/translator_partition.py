"""Partition metrics for Translator event representations."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def _as_feature_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"features must be a 2D matrix, got shape {x.shape}")
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def balanced_sample_indices(
    labels: Sequence[Any],
    *,
    max_events: int | None,
    seed: int,
) -> np.ndarray:
    """Return a balanced subsample of row indices, stratified by label."""
    labels = np.asarray(labels, dtype=object)
    n = labels.size
    if max_events is None or int(max_events) <= 0 or n <= int(max_events):
        return np.arange(n, dtype=np.int64)

    max_events = int(max_events)
    rng = np.random.default_rng(seed)
    label_strings = labels.astype(str)
    groups: list[np.ndarray] = []
    for value in pd.unique(label_strings):
        groups.append(np.flatnonzero(label_strings == value))

    per_group = max(1, max_events // len(groups))
    selected: list[np.ndarray] = []
    leftovers: list[np.ndarray] = []
    for idx in groups:
        if idx.size <= per_group:
            selected.append(idx)
        else:
            take = rng.choice(idx, size=per_group, replace=False)
            selected.append(take)
            leftovers.append(np.setdiff1d(idx, take, assume_unique=False))

    selected_idx = np.concatenate(selected)
    if selected_idx.size > max_events:
        selected_idx = rng.choice(selected_idx, size=max_events, replace=False)
    remaining = max_events - selected_idx.size
    if remaining > 0 and leftovers:
        pool = np.concatenate([x for x in leftovers if x.size])
        if pool.size:
            extra = rng.choice(pool, size=min(remaining, pool.size), replace=False)
            selected_idx = np.concatenate([selected_idx, extra])
    return np.sort(selected_idx).astype(np.int64)


def metric_representation(
    features: np.ndarray,
    *,
    n_pcs: int,
    seed: int,
) -> np.ndarray:
    """Standardize features and optionally reduce dimensionality with PCA."""
    x = _as_feature_matrix(features)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    if x.shape[1] <= n_pcs or x.shape[0] <= n_pcs + 1:
        return x.astype(np.float32, copy=False)
    n_comps = max(2, min(int(n_pcs), x.shape[1], x.shape[0] - 1))
    return PCA(n_components=n_comps, random_state=seed).fit_transform(x).astype(np.float32)


def compute_partition_metrics(
    features: np.ndarray,
    meta: pd.DataFrame,
    louvain: np.ndarray,
    *,
    label_columns: Sequence[str],
    max_events: int,
    knn_k: int,
    n_pcs: int,
    seed: int,
) -> pd.DataFrame:
    """Compute clustering/partition quality metrics for event embeddings."""
    rows: list[dict[str, Any]] = []
    x_all = metric_representation(features, n_pcs=n_pcs, seed=seed)
    for label_col in label_columns:
        if label_col not in meta:
            continue
        labels_all = meta[label_col].astype(str).to_numpy()
        valid = labels_all != ""
        valid &= labels_all != "nan"
        valid &= labels_all != "unknown"
        if valid.sum() < 3:
            continue

        idx_valid = np.flatnonzero(valid)
        idx_eval = idx_valid[
            balanced_sample_indices(labels_all[idx_valid], max_events=max_events, seed=seed)
        ]
        x = x_all[idx_eval]
        labels = labels_all[idx_eval]
        clusters = np.asarray(louvain, dtype=str)[idx_eval]

        codes, uniques = pd.factorize(labels)
        n_classes = len(uniques)
        if n_classes < 2 or labels.size <= n_classes:
            continue

        sample_size = min(10000, labels.size)
        metric_rows = [
            (
                "silhouette",
                float(
                    silhouette_score(
                        x,
                        codes,
                        sample_size=sample_size,
                        random_state=seed,
                    )
                ),
            )
        ]
        n_neighbors = min(max(2, int(knn_k) + 1), labels.size)
        nn = NearestNeighbors(n_neighbors=n_neighbors).fit(x).kneighbors(
            x,
            return_distance=False,
        )[:, 1:]
        metric_rows.append(("knn_purity", float(np.mean(labels[nn] == labels[:, None]))))

        if label_col == "gt_concept":
            cluster_codes, _ = pd.factorize(clusters)
            metric_rows.extend(
                (
                    (
                        "cluster_nmi",
                        float(normalized_mutual_info_score(codes, cluster_codes)),
                    ),
                    (
                        "cluster_ari",
                        float(adjusted_rand_score(codes, cluster_codes)),
                    ),
                )
            )

        rows.extend(
            {
                "label": label_col,
                "metric": metric,
                "value": value,
            }
            for metric, value in metric_rows
        )
    return pd.DataFrame(rows)
