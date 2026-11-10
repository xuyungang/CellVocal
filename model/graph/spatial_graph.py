"""Step 3: spatial neighbourhood graph G_sp = (V, E_sp)."""
from __future__ import annotations

from typing import Optional

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.neighbors import kneighbors_graph

from model.train.config import CellVOCALGraphConfig


def build_spatial_graph(
    adata: ad.AnnData,
    config: CellVOCALGraphConfig,
    *,
    coord_key: Optional[str] = None,
) -> sparse.csr_matrix:
    """Build spatial connectivities and store in ``adata.obsp[config.spatial_graph_key]``."""
    key = coord_key or config.spatial_coord_key
    x = np.asarray(adata.obsm[key], dtype=np.float64)
    n = min(config.n_spatial_neighbors, max(1, adata.n_obs - 1))
    g = kneighbors_graph(x, n_neighbors=n, mode="distance", include_self=config.self_loops)
    g = g.maximum(g.T).tocsr()
    adata.obsp[config.spatial_graph_key] = g
    return g
