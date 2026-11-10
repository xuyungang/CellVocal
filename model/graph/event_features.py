"""Frozen kinetic-state helpers for translator pretraining."""
from __future__ import annotations

import numpy as np
import torch


def frozen_kinetic_states(
    unspliced: np.ndarray,
    spliced: np.ndarray,
    *,
    latent_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.cat(
        [
            torch.as_tensor(unspliced, device=device, dtype=torch.float32),
            torch.as_tensor(spliced, device=device, dtype=torch.float32),
        ],
        dim=-1,
    )
    if x.shape[1] >= latent_dim:
        z = x[:, :latent_dim]
    else:
        z = torch.nn.functional.pad(x, (0, latent_dim - x.shape[1]))
    return z.detach(), z.norm(dim=1, keepdim=True).detach()
