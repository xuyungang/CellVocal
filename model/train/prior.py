"""Dataset-prior preprocessing shared by all CellVOCAL training presets."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def smooth_graph_signal(
    values: np.ndarray,
    graph: sparse.spmatrix,
    *,
    steps: int,
    alpha: float,
) -> np.ndarray:
    """Diffuse a scalar prior over the spatial graph without using labels."""
    signal = np.asarray(values, dtype=np.float64).reshape(-1)
    n_steps = max(0, int(steps))
    if n_steps == 0:
        return signal.copy()
    if graph.shape != (signal.size, signal.size):
        raise ValueError(
            f"graph shape {graph.shape} does not match prior length {signal.size}"
        )
    mix = float(alpha)
    if not 0.0 <= mix <= 1.0:
        raise ValueError("time_prior_smoothing_alpha must lie in [0, 1]")

    adjacency = graph.tocsr(copy=True).astype(np.float64)
    adjacency.data.fill(1.0)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    valid = degree > 0.0
    for _ in range(n_steps):
        neighbour_mean = signal.copy()
        neighbour_mean[valid] = (
            np.asarray(adjacency @ signal).ravel()[valid] / degree[valid]
        )
        signal = (1.0 - mix) * signal + mix * neighbour_mean
    return signal
