from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

SPATIAL_KNN_KEY = "X_spatial_knn"
_N_SPATIAL_NEIGHBORS = 30
_SPATIAL_GRAPH_K = 16
_KNN_JITTER = 1e-5

_K_CBDIR = ("K-CBDir", "K-CBDir (Gene Space)")
_PALETTE = ("#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#76B7B2")
_MARKERS = ("o", "s", "^", "D", "v")

_SCVELO_PARALLEL_PATCHED = False


def _patch_sparse() -> None:
    """TopoVelo evaluation still uses sparse ``.A``; restore on newer SciPy."""
    if hasattr(sparse.csr_matrix, "A"):
        return
    for cls in (
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.bsr_matrix,
        sparse.coo_matrix,
        sparse.dok_matrix,
        sparse.lil_matrix,
    ):
        cls.A = property(lambda self: self.toarray())  # type: ignore[attr-defined]
        cls.A1 = property(lambda self: self.toarray().ravel())  # type: ignore[attr-defined]


def _patch_scvelo_parallel() -> None:
    """scVelo 0.2.x + NumPy 2.x: parallelize(as_array=True) breaks velocity_graph."""
    global _SCVELO_PARALLEL_PATCHED
    if _SCVELO_PARALLEL_PATCHED:
        return

    from scvelo.core import get_n_jobs
    from scvelo.core._parallelize import parallelize
    from scvelo.tools.velocity_graph import VelocityGraph, _flatten, vals_to_csr

    _orig_compute_cosines_cell = VelocityGraph._compute_cosines

    def _compute_cosines(self, obs_idx, queue):
        # All-NaN velocity → scVelo drops every gene → V has shape (n_obs, 0).
        if getattr(self, "V", None) is None or self.V.size == 0 or self.V.shape[1] == 0:
            if queue is not None:
                queue.put(None)
            return [], [], [], []
        return _orig_compute_cosines_cell(self, obs_idx, queue)

    def compute_cosines(
        self,
        n_jobs=None,
        backend="loky",
        show_progress_bar: bool = True,
        **_kwargs,
    ):
        n_jobs = get_n_jobs(n_jobs=n_jobs)
        n_obs = self.X.shape[0]
        if getattr(self, "V", None) is None or self.V.size == 0 or self.V.shape[1] == 0:
            self.graph = sparse.csr_matrix((n_obs, n_obs), dtype=np.float64)
            self.graph_neg = sparse.csr_matrix((n_obs, n_obs), dtype=np.float64)
            self.uncertainties = None
            self.self_prob = np.zeros(n_obs, dtype=np.float64)
            return

        res = parallelize(
            self._compute_cosines,
            range(self.X.shape[0]),
            n_jobs=n_jobs,
            unit="cells",
            backend=backend,
            as_array=False,
            show_progress_bar=show_progress_bar,
        )()
        uncertainties, vals, rows, cols = map(_flatten, zip(*res))

        if len(vals) == 0:
            self.graph = sparse.csr_matrix((n_obs, n_obs), dtype=np.float64)
            self.graph_neg = sparse.csr_matrix((n_obs, n_obs), dtype=np.float64)
            self.uncertainties = None
            self.self_prob = np.zeros(n_obs, dtype=np.float64)
            return

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals,
            rows,
            cols,
            shape=(n_obs, n_obs),
            split_negative=True,
        )
        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties,
                rows,
                cols,
                shape=(n_obs, n_obs),
                split_negative=False,
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

    VelocityGraph._compute_cosines = _compute_cosines  # type: ignore[method-assign]
    VelocityGraph.compute_cosines = compute_cosines  # type: ignore[method-assign]
    _SCVELO_PARALLEL_PATCHED = True


def _patch_runtime() -> None:
    _patch_sparse()
    _patch_scvelo_parallel()


def spatial_graph(adata: ad.AnnData, *, k: int = _SPATIAL_GRAPH_K) -> None:
    """TopoVelo simulation spatial graph on ``obsm['X_spatial']``."""
    x = np.asarray(adata.obsm["X_spatial"], dtype=np.float64)
    g = NearestNeighbors(n_neighbors=k).fit(x).kneighbors_graph()
    adata.obsp["spatial_graph"] = sparse.csr_matrix(g, dtype=np.float32)


def embed_spatial_velocity(
    adata: ad.AnnData,
    velocity_key: str,
    *,
    n_jobs: int | None = None,
    spatial_key: str = "spatial",
    inplace: bool = False,
) -> np.ndarray:
    """Project ``layers[f'{velocity_key}_velocity']`` into spatial coordinates.

    Returns the 2D field. When ``inplace`` is True, also stores it as
    ``obsm[f'{velocity_key}_velocity_spatial']``.
    """
    _patch_runtime()
    vkey = f"{velocity_key}_velocity"
    out_key = f"{vkey}_spatial"
    if vkey not in adata.layers:
        raise KeyError(f"Missing layers[{vkey!r}]")
    if spatial_key not in adata.obsm and "X_spatial" not in adata.obsm:
        raise KeyError(f"Missing obsm[{spatial_key!r}]")

    gene_velocity = np.asarray(adata.layers[vkey], dtype=np.float64)
    if not np.isfinite(gene_velocity).any():
        velocity = np.zeros((adata.n_obs, 2), dtype=np.float64)
        if inplace:
            adata.obsm[out_key] = velocity
        return velocity

    work = adata if inplace else adata.copy()
    work.obsm["X_spatial"] = np.asarray(
        work.obsm["X_spatial"] if "X_spatial" in work.obsm else work.obsm[spatial_key],
        dtype=np.float64,
    )[:, :2]
    jobs = 1 if n_jobs is None or n_jobs <= 0 else int(n_jobs)
    scv.tl.velocity_graph(work, vkey=vkey, n_jobs=jobs)
    scv.tl.velocity_embedding(work, vkey=vkey, basis="spatial")
    return np.asarray(work.obsm[out_key], dtype=np.float64)


def prepare_metrics_adata(
    adata: ad.AnnData,
    velocity_key: str,
    *,
    seed: int | None = None,
) -> ad.AnnData:
    """
    AnnData prep for TopoVelo ``post_analysis`` metrics.

    ``seed`` set → simulation benchmark (``X_spatial_knn``); otherwise tissue (``spatial``).
    """
    _patch_runtime()
    prepared = adata.copy()
    if seed is not None:
        x = np.asarray(prepared.obsm["X_spatial"], dtype=np.float64)
        scale = max(float(np.abs(x).max()), float(np.ptp(x)), 1.0)
        rng = np.random.default_rng(seed)
        prepared.obsm[SPATIAL_KNN_KEY] = x + (_KNN_JITTER * scale) * rng.standard_normal(x.shape)
        sc.pp.neighbors(
            prepared,
            use_rep=SPATIAL_KNN_KEY,
            n_neighbors=max(1, min(_N_SPATIAL_NEIGHBORS, prepared.n_obs - 1)),
        )
    spatial_graph(prepared)
    if "Ms" not in prepared.layers or "Mu" not in prepared.layers:
        if seed is None:
            prepared.obsm["_neighbor_rep"] = np.asarray(
                prepared.layers[f"{velocity_key}_velocity"], dtype=np.float64
            )
            sc.pp.neighbors(
                prepared,
                use_rep="_neighbor_rep",
                n_neighbors=max(1, min(_N_SPATIAL_NEIGHBORS, prepared.n_obs - 1)),
                n_pcs=min(30, prepared.n_vars),
            )
            del prepared.obsm["_neighbor_rep"]
        scv.pp.moments(prepared, n_pcs=30, n_neighbors=_N_SPATIAL_NEIGHBORS)
    return prepared


class VelocityEvaluator:
    """RNA velocity benchmark metrics via TopoVelo ``post_analysis`` + ``PerfLogger``."""

    SUMMARY_METRICS = (
        "Time Correlation",
        "Spatial Velocity Consistency",
        "Spatial Time Consistency",
        "CBDir",
    )

    def __init__(
        self,
        perf_dir: str | Path,
        *,
        spatial_graph_key: str = "spatial_graph",
        spatial_key: str = "spatial",
        cluster_key: str = "clusters",
        n_spatial_neighbors: int = _N_SPATIAL_NEIGHBORS,
    ) -> None:
        # TopoVelo pulls in the full torch-geometric/transformers stack. Keep
        # that optional dependency lazy so plotting-only analysis can start
        # without paying its import and filesystem-scan cost.
        import topovelo as tpv

        self._tpv = tpv
        self.perf_dir = Path(perf_dir)
        self.perf_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_graph_key = spatial_graph_key
        self.spatial_key = spatial_key
        self.cluster_key = cluster_key
        self.n_spatial_neighbors = n_spatial_neighbors
        self._logger = self._tpv.PerfLogger(save_path=str(self.perf_dir))

    @staticmethod
    def chain_edges(n_clusters: int, start: int = 0) -> list[tuple[str, str]]:
        """Ground-truth transitions ``start→start+1→…`` (e.g. bilinear Leiden 0–4)."""
        if n_clusters < 2:
            return []
        return [(str(start + i), str(start + i + 1)) for i in range(n_clusters - 1)]

    def evaluate(
        self,
        adata: ad.AnnData,
        dataset_name: str,
        test_id: str,
        methods: Sequence[str],
        keys: Sequence[str],
        cluster_edges: Sequence[tuple[str, str]],
        *,
        spatial_velocity_graph: bool = False,
        n_gene_plots: int = 0,
        seed: int | None = None,
        n_jobs: int | None = None,
        has_time_prediction: bool = True,
    ) -> None:
        """Compute metrics for one dataset, append to perf log, save K-CBDir figure."""
        _patch_runtime()
        genes: list[str] = []
        if n_gene_plots > 0:
            rng = np.random.default_rng(seed)
            n = min(n_gene_plots, adata.n_vars)
            genes = list(rng.choice(adata.var_names, size=n, replace=False))

        jobs = 1 if n_jobs is None or n_jobs <= 0 else int(n_jobs)
        method_list = list(methods)
        key_list = list(keys)

        velocity_ok = True
        work = adata.copy()
        for key in key_list:
            vkey = f"{key}_velocity"
            if vkey not in work.layers:
                velocity_ok = False
                continue
            vel = np.asarray(work.layers[vkey], dtype=np.float64)
            if not np.isfinite(vel).any():
                velocity_ok = False
                # Keep gene columns finite so scVelo does not drop to a (n, 0) velocity matrix.
                work.layers[vkey] = np.zeros_like(vel, dtype=np.float64)
            else:
                work.layers[vkey] = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)

        if not velocity_ok:
            # Avoid TopoVelo pairwise-metric shape errors on empty/degenerate graphs.
            nan_metrics = list(self.SUMMARY_METRICS) + [
                "CBDir (Gene Space)",
                "Time Score",
                "Velocity Consistency",
            ]
            for method in method_list:
                for metric in nan_metrics:
                    self._logger.df.loc[(metric, method), dataset_name] = np.nan
            self._plot_k_cbdir(dataset_name, method_list)
            return

        result = self._tpv.post_analysis(
            work,
            test_id,
            method_list,
            key_list,
            n_jobs=jobs,
            spatial_graph_key=self.spatial_graph_key,
            spatial_key=self.spatial_key,
            spatial_velocity_graph=spatial_velocity_graph,
            n_spatial_neighbors=self.n_spatial_neighbors,
            compute_metrics=True,
            genes=genes,
            grid_size=(1, max(len(genes), 1)),
            embed="spatial",
            plot_type=[],
            cluster_key=self.cluster_key,
            cluster_edges=list(cluster_edges),
        )
        if result[0] is None:
            raise RuntimeError(
                "post_analysis failed (check spatial graph, clusters, velocity layers)."
            )

        self._logger.insert(dataset_name, *result)
        if not has_time_prediction:
            for method in method_list:
                for metric in (
                    "Time Correlation",
                    "Spatial Time Consistency",
                    "Time Score",
                ):
                    index = (metric, method)
                    if index in self._logger.df.index:
                        self._logger.df.loc[index, dataset_name] = np.nan
        self._plot_k_cbdir(dataset_name, method_list)

    def _plot_k_cbdir(self, dataset_name: str, methods: Sequence[str]) -> None:
        df_multi = self._logger.df_multi
        if dataset_name not in df_multi.columns.get_level_values(0):
            return

        metrics = [m for m in _K_CBDIR if m in df_multi.index.get_level_values(0)]
        if not metrics:
            return

        method_list = list(methods) or list(df_multi.index.get_level_values(1).unique())
        steps = list(df_multi.columns.get_level_values(1).unique())
        if not steps:
            return

        x = np.arange(1, len(steps) + 1)
        slug = "_".join(re.sub(r"\W+", " ", dataset_name.lower()).rstrip().split())
        fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 2.2), squeeze=False)

        for ax, metric in zip(axes.ravel(), metrics):
            for i, method in enumerate(method_list):
                if (metric, method) not in df_multi.index:
                    continue
                y = np.asarray(df_multi.loc[(metric, method), dataset_name].values, dtype=float)
                if y.size == 0 or np.all(np.isnan(y)):
                    continue
                ax.plot(
                    x[: y.size],
                    y,
                    color=_PALETTE[i % len(_PALETTE)],
                    marker=_MARKERS[i % len(_MARKERS)],
                    markersize=3.5,
                    linewidth=0.9,
                    label=method,
                )
            ax.set_title(metric, fontsize=7)
            ax.set_xlabel("Step size", fontsize=7)
            ax.set_ylabel(metric, fontsize=7)
            ax.set_xticks(x, [str(v) for v in x])
            ax.tick_params(labelsize=6)

        if len(method_list) > 1:
            handles, labels = axes.ravel()[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper center", ncol=min(len(method_list), 4), fontsize=6)

        fig.tight_layout()
        fig.savefig(self.perf_dir / f"{slug}_k_cbdir.pdf", dpi=300, bbox_inches="tight")
        plt.close(fig)

    def save(self, prefix: str = "simulation") -> None:
        self._logger.save(prefix)

    def summary(self) -> pd.DataFrame:
        return self._logger.df.loc[list(self.SUMMARY_METRICS)]
