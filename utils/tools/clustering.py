"""Clustering utilities: PCA, Louvain, and UMAP layout."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from typing import Optional, Tuple, Union

import anndata as ad
import igraph as ig
import leidenalg
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.decomposition import PCA

_BASE_SEED = 2022


def active_base_seed() -> int:
    return _BASE_SEED


def pca(adata: ad.AnnData, use_reps: Optional[str] = None, n_comps: int = 10) -> np.ndarray:
    data = adata.obsm[use_reps] if use_reps else adata.X
    if issparse(data):
        data = data.toarray()
    data = np.asarray(data, dtype=np.float32)
    n_samples, n_features = data.shape
    n_comps = max(1, min(n_comps, n_features, n_samples - 1))
    return PCA(n_components=n_comps, random_state=active_base_seed()).fit_transform(data)


def _igraph_from_connectivities(adj) -> ig.Graph:
    mat = csr_matrix(adj)
    mat = mat.maximum(mat.T).tocoo()
    mask = mat.row < mat.col
    if not np.any(mask):
        raise ValueError("connectivity graph has no edges")
    edges = list(zip(mat.row[mask].tolist(), mat.col[mask].tolist()))
    graph = ig.Graph(n=mat.shape[0], edges=edges, directed=False)
    graph.es["weight"] = mat.data[mask].astype(float).tolist()
    return graph


def _louvain_at_resolution(adata: ad.AnnData, resolution: float) -> np.ndarray:
    graph = _igraph_from_connectivities(adata.obsp["connectivities"])
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=2,
        seed=active_base_seed(),
    )
    return np.asarray(partition.membership, dtype=int)


def _louvain_search(
    adata: ad.AnnData,
    n_clusters: int,
    *,
    start_res: float = 0.01,
    end_res: float = 2,
    increment: float = 0.001,
) -> np.ndarray:
    lo, hi = float(start_res), float(end_res)
    tol = max(float(increment), 1e-6)

    def _eval(res: float) -> tuple[np.ndarray, int]:
        labels = _louvain_at_resolution(adata, res)
        return labels, int(len(np.unique(labels)))

    lo_labels, lo_count = _eval(lo)
    hi_labels, hi_count = _eval(hi)
    if n_clusters < lo_count or n_clusters > hi_count:
        raise ValueError(
            f"target n_clusters={n_clusters} outside achievable range [{lo_count}, {hi_count}]"
        )
    if lo_count == n_clusters:
        return lo_labels
    if hi_count == n_clusters:
        return hi_labels

    best_diff, best_labels = abs(lo_count - n_clusters), lo_labels
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        labels, count = _eval(mid)
        if count == n_clusters:
            return labels
        diff = abs(count - n_clusters)
        if diff < best_diff:
            best_diff, best_labels = diff, labels
        if count > n_clusters:
            hi = mid
        else:
            lo = mid

    return best_labels


def _build_neighbor_adata(
    features: Union[torch.Tensor, np.ndarray],
    obs_names,
    *,
    n_neighbors: int,
    n_comps: int,
    use_pca: bool = True,
) -> ad.AnnData:
    features = np.asarray(features, dtype=np.float32)
    adata = ad.AnnData(features)
    adata.obs_names = obs_names
    n_samples, n_features = features.shape
    n_comps = max(2, min(n_comps, n_features, n_samples - 1))
    n_neighbors = min(n_neighbors, n_samples - 1)
    if use_pca:
        adata.obsm["X_pca"] = pca(adata, use_reps=None, n_comps=n_comps)
        use_rep = "X_pca"
    else:
        adata.obsm["_rep"] = features
        use_rep = "_rep"
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        random_state=active_base_seed(),
    )
    return adata


def _umap_embed(adata: ad.AnnData) -> np.ndarray:
    sc.tl.umap(adata)
    return np.asarray(adata.obsm["X_umap"], dtype=np.float32)


class ClusteringUtils:
    """Louvain clustering and UMAP layout on a shared neighbor graph."""

    @staticmethod
    def auto_n_clusters(n_samples: int, min_clusters: int = 2, max_clusters: int = 15) -> int:
        return max(min_clusters, min(int(np.sqrt(n_samples)), max_clusters))

    @staticmethod
    def cluster_and_umap(
        features: Union[torch.Tensor, np.ndarray],
        obs_names,
        n_clusters: Optional[int] = None,
        n_neighbors: int = 50,
        n_comps: int = 20,
        use_pca: bool = True,
        start_res: float = 0.01,
        end_res: float = 2,
        increment: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        n_clusters = (
            ClusteringUtils.auto_n_clusters(features.shape[0])
            if n_clusters is None
            else int(n_clusters)
        )
        adata = _build_neighbor_adata(
            features,
            obs_names,
            n_neighbors=n_neighbors,
            n_comps=n_comps,
            use_pca=use_pca,
        )
        labels = _louvain_search(
            adata,
            n_clusters,
            start_res=start_res,
            end_res=end_res,
            increment=increment,
        )
        umap_xy = _umap_embed(adata)
        return umap_xy, labels
