"""Vectorized per-group reductions (avoid O(n_groups * n_events) boolean masks)."""
from __future__ import annotations

import torch


def group_index(key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return ``(unique_keys, inverse_index, n_groups)`` with dense ids ``0..n_groups-1``."""
    uniq, inv = torch.unique(key, return_inverse=True)
    n_groups = int(uniq.numel())
    return uniq, inv, n_groups


def group_mean(values: torch.Tensor, inv: torch.Tensor, n_groups: int) -> torch.Tensor:
    out_shape = (n_groups,) + tuple(values.shape[1:])
    sums = torch.zeros(out_shape, device=values.device, dtype=values.dtype)
    index = inv
    if values.ndim > 1:
        index = inv.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    sums.scatter_add_(0, index, values)
    counts = torch.zeros(n_groups, device=values.device, dtype=values.dtype)
    counts.scatter_add_(
        0,
        inv,
        torch.ones(values.shape[0], device=values.device, dtype=values.dtype),
    )
    view_shape = (n_groups,) + (1,) * (values.ndim - 1)
    return sums / counts.clamp_min(1.0).view(view_shape)


def group_sum(values: torch.Tensor, inv: torch.Tensor, n_groups: int) -> torch.Tensor:
    out_shape = (n_groups,) + tuple(values.shape[1:])
    out = torch.zeros(out_shape, device=values.device, dtype=values.dtype)
    index = inv
    if values.ndim > 1:
        index = inv.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    out.scatter_add_(0, index, values)
    return out
