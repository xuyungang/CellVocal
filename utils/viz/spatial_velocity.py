"""Spatial velocity rendering shared with Extended Data Fig. 1a.

Vectors are averaged in occupied bins, then normalized to equal-length arrows.
Only the scatter background is subsampled; every finite cell contributes to
the grid average. Arrow length encodes direction, not velocity magnitude.
"""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CELL_TYPE_PALETTE = (
    "#007C83",
    "#E69F00",
    "#4E79A7",
    "#59A14F",
    "#CC79A7",
    "#D55E4F",
    "#9467BD",
    "#8C564B",
    "#17BECF",
    "#BCBD22",
    "#F28E2B",
    "#76B7B2",
    "#B07AA1",
    "#EDC948",
    "#BAB0AC",
    "#6B6ECF",
)


def normalize_time(values: np.ndarray) -> np.ndarray:
    """Min-max normalize finite times without quantile clipping."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def spatial_sample(xy: np.ndarray, max_cells: int, *, seed: int) -> np.ndarray:
    """Sample occupied spatial bins first, then fill the remaining cell budget."""
    n_obs = xy.shape[0]
    if max_cells <= 0 or n_obs <= max_cells:
        return np.arange(n_obs, dtype=np.int64)

    rng = np.random.default_rng(seed)
    side = max(8, int(np.ceil(np.sqrt(max_cells / 1.5))))
    mins = np.nanmin(xy, axis=0)
    spans = np.maximum(np.nanmax(xy, axis=0) - mins, 1e-12)
    bins = np.floor((xy - mins) / spans * side).astype(np.int64)
    bins = np.clip(bins, 0, side - 1)
    flat = bins[:, 1] * side + bins[:, 0]

    selected: list[int] = []
    for bin_id in np.unique(flat):
        members = np.flatnonzero(flat == bin_id)
        selected.append(int(rng.choice(members)))

    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.size >= max_cells:
        return np.sort(rng.choice(selected_array, max_cells, replace=False))

    remaining = np.setdiff1d(
        np.arange(n_obs, dtype=np.int64),
        selected_array,
        assume_unique=False,
    )
    n_extra = min(max_cells - selected_array.size, remaining.size)
    extra = rng.choice(remaining, n_extra, replace=False)
    return np.sort(np.concatenate([selected_array, extra]))


def grid_vectors(
    xy: np.ndarray,
    velocity: np.ndarray,
    time_values: np.ndarray,
    *,
    grid_size: int,
    min_cells: int,
    velocity_prefix: str,
    time_column: str,
) -> pd.DataFrame:
    """Average cell positions and velocities in sufficiently populated bins."""
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if min_cells < 1:
        raise ValueError("min_cells must be positive.")

    mins = np.nanmin(xy, axis=0)
    maxs = np.nanmax(xy, axis=0)
    edges_x = np.linspace(mins[0], maxs[0], grid_size + 1)
    edges_y = np.linspace(mins[1], maxs[1], grid_size + 1)
    ix = np.clip(np.digitize(xy[:, 0], edges_x) - 1, 0, grid_size - 1)
    iy = np.clip(np.digitize(xy[:, 1], edges_y) - 1, 0, grid_size - 1)
    flat = iy * grid_size + ix

    rows: list[dict[str, Any]] = []
    for bin_id in np.unique(flat):
        members = np.flatnonzero(flat == bin_id)
        if members.size < min_cells:
            continue
        mean_velocity = np.nanmean(velocity[members], axis=0)
        rows.append(
            {
                "grid_x": int(bin_id % grid_size),
                "grid_y": int(bin_id // grid_size),
                "x": float(np.nanmean(xy[members, 0])),
                "y": float(np.nanmean(xy[members, 1])),
                "n_cells": int(members.size),
                time_column: float(np.nanmean(time_values[members])),
                f"{velocity_prefix}_vx": float(mean_velocity[0]),
                f"{velocity_prefix}_vy": float(mean_velocity[1]),
                f"{velocity_prefix}_speed": float(np.linalg.norm(mean_velocity)),
            }
        )
    return pd.DataFrame(rows, columns=[
        "grid_x", "grid_y", "x", "y", "n_cells", time_column,
        f"{velocity_prefix}_vx", f"{velocity_prefix}_vy",
        f"{velocity_prefix}_speed",
    ])


def unit_vectors(vectors: np.ndarray, *, arrow_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalize nonzero finite mean vectors to a common display length."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    keep = np.isfinite(vectors).all(axis=1) & (norms > 1e-12)
    unit = np.zeros_like(vectors)
    unit[keep] = vectors[keep] / norms[keep, None] * arrow_length
    return keep, unit


def map_limits(xy: np.ndarray) -> tuple[float, float, float, float]:
    x_span = max(float(np.ptp(xy[:, 0])), 1e-12)
    y_span = max(float(np.ptp(xy[:, 1])), 1e-12)
    pad_x = 0.035 * x_span
    pad_y = 0.035 * y_span
    return (
        float(np.min(xy[:, 0]) - pad_x),
        float(np.max(xy[:, 0]) + pad_x),
        float(np.min(xy[:, 1]) - pad_y),
        float(np.max(xy[:, 1]) + pad_y),
    )


def style_spatial_axis(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_velocity_arrows(
    ax: plt.Axes,
    xy: np.ndarray,
    grid: pd.DataFrame,
    vector_columns: tuple[str, str],
) -> None:
    """Draw the reference panel's direction arrows; empty grids draw nothing."""
    if grid.empty:
        return
    x_span = max(float(np.ptp(xy[:, 0])), 1e-12)
    y_span = max(float(np.ptp(xy[:, 1])), 1e-12)
    arrow_length = 0.62 * min(
        x_span / max(int(grid["grid_x"].max()) + 1, 1),
        y_span / max(int(grid["grid_y"].max()) + 1, 1),
    )
    vectors = grid.loc[:, list(vector_columns)].to_numpy(dtype=float)
    keep, arrows = unit_vectors(vectors, arrow_length=arrow_length)
    ax.quiver(
        grid.loc[keep, "x"],
        grid.loc[keep, "y"],
        arrows[keep, 0],
        arrows[keep, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        pivot="mid",
        color="#202020",
        width=0.0048,
        headwidth=3.2,
        headlength=4.0,
        headaxislength=3.6,
        alpha=0.90,
        zorder=2,
    )
