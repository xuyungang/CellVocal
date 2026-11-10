from __future__ import annotations

import numpy as np
import topovelo as tpv

TMAX = 20.0


def set_real_time_prior(adata, dataset_name: str) -> np.ndarray:
    """Store the shared real-tissue time prior in ``obs['tprior']``."""
    if "X_spatial" not in adata.obsm:
        if "spatial" not in adata.obsm:
            raise KeyError("Real-tissue time prior requires spatial coordinates.")
        adata.obsm["X_spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    if dataset_name == "mouse_brain_cortex":
        x = -np.asarray(adata.obsm["X_spatial"][:, 0], dtype=np.float64)
        span = float(np.ptp(x))
        if span == 0:
            raise ValueError("Cannot derive cortex time prior from a constant x coordinate.")
        adata.obs["tprior"] = TMAX * (x - x.min()) / span
    elif dataset_name in {"mouse_gut", "mouse_lung"}:
        tpv.get_spatial_tprior(adata, tmax=TMAX, basis="spatial", q=0.95)
    else:
        raise ValueError(f"Unsupported real-tissue dataset: {dataset_name!r}")

    prior = np.asarray(adata.obs["tprior"], dtype=np.float64)
    if prior.shape != (adata.n_obs,) or not np.all(np.isfinite(prior)):
        raise ValueError("Time prior must contain one finite value per cell.")
    if float(np.ptp(prior)) == 0:
        raise ValueError("Time prior is constant.")
    return prior


def set_time_prior_endpoints(
    adata,
    *,
    root_key: str = "time_prior_root",
    end_key: str = "time_prior_end",
) -> tuple[str, str]:
    """Mark the earliest/latest 5% of ``tprior`` for scVelo latent time."""
    prior = np.asarray(adata.obs["tprior"], dtype=np.float64)
    adata.obs[root_key] = (prior <= np.quantile(prior, 0.05)).astype(np.float32)
    adata.obs[end_key] = (prior >= np.quantile(prior, 0.95)).astype(np.float32)
    return root_key, end_key
