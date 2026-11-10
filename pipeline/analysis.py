"""Post-training analysis utilities for CellVOCAL (thin facade over shared tools)."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd

from model.train.config import CellVOCALConfig
from utils.metric import CCCRecoveryEvaluator, VelocityEvaluator
from utils.metric.velocity import SPATIAL_KNN_KEY, prepare_metrics_adata
from utils.viz.lr_spatial import LRSpatialPlotter, parse_lr_pairs
from utils.viz.translator import TranslatorUmapTransfer
from utils.viz.velocity import VelocityEmbeddingPlotter


class CellVOCALAnalysis:
    """
    External facade for post-training analysis.

    Delegates UMAP transfer to :class:`TranslatorUmapTransfer`; plotting uses
    shared viz helpers under ``utils.viz``.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        config: Optional[CellVOCALConfig] = None,
    ) -> None:
        self.adata = adata
        self.config = config or CellVOCALConfig()

    @classmethod
    def from_h5ad(
        cls,
        path: Union[str, Path],
        config: Optional[CellVOCALConfig] = None,
    ) -> CellVOCALAnalysis:
        return cls(ad.read_h5ad(path), config=config)

    def communication_events(self, *, adata=None) -> pd.DataFrame:
        """Validated sender/receiver/module events, including q and sender fractions."""
        from utils.tools.data import communication_events
        return communication_events(
            self.adata if adata is None else adata, prefix=self.config.eval.velocity_key)

    def incoming_influence(self, *, adata=None) -> pd.DataFrame:
        """Receiver-by-module incoming influence matrix; indices follow the full fitted AnnData."""
        from utils.tools.data import incoming_influence
        return incoming_influence(
            self.adata if adata is None else adata, prefix=self.config.eval.velocity_key)

    def program_scores(self, genes: Sequence[str], *, adata=None, **kwargs) -> pd.DataFrame:
        """Project expression and RNA velocities onto a shared gene signature.

        See :func:`utils.tools.data.program_scores` for gene-scale
        and missing-gene options. The original AnnData is never modified.
        """
        from utils.tools.data import program_scores
        return program_scores(self.adata if adata is None else adata, genes,
                              prefix=self.config.eval.velocity_key, **kwargs)

    def response_components(self, *, adata=None, checkpoint_path=None, batch_size=1024):
        """Read decoder components from the fitted checkpoint, without retraining."""
        from utils.tools.data import load_response_components
        path = (Path(self.config.train.checkpoint_dir) / "cellvocal.pt"
                if checkpoint_path is None else checkpoint_path)
        return load_response_components(
            self.adata if adata is None else adata, path,
            prefix=self.config.eval.velocity_key, batch_size=batch_size)

    @staticmethod
    def prepare_communication_counts(events, sample_metadata, **kwargs) -> pd.DataFrame:
        """Build zero-filled sample/group event counts with retained-event offsets."""
        from utils.tools.regression import prepare_event_counts
        return prepare_event_counts(events, sample_metadata, **kwargs)

    @staticmethod
    def communication_regression(counts, **kwargs) -> pd.DataFrame:
        """Negative-binomial clinical contrasts with donor-clustered covariance.

        See :func:`utils.tools.regression.fit_communication_regression`.
        This analysis requires multiple independent donors, not individual spots
        treated as replicates.
        """
        from utils.tools.regression import fit_communication_regression
        return fit_communication_regression(counts, **kwargs)

    @staticmethod
    def plot_communication_regression(results, **kwargs):
        """Plot regression log2 rate ratios; return a Matplotlib figure and axes."""
        from utils.viz.regression import plot_communication_regression
        return plot_communication_regression(results, **kwargs)

    def screen_communication(self, outcome, *, incoming=None, adata=None, **kwargs):
        """Candidate screen: input/outcome coupling versus high/background contrast.

        Returns per-section and equal-section-weighted summary DataFrames.
        ``outcome`` explicitly chooses total or communication-associated dynamics.
        See :func:`utils.tools.screen.screen_communication_modules`.
        """
        from utils.tools.screen import screen_communication_modules
        if incoming is None:
            incoming = self.incoming_influence(adata=adata)
        return screen_communication_modules(incoming, outcome, **kwargs)

    @staticmethod
    def plot_communication_screen(summary, **kwargs):
        """Panel-B-style candidate bubble plot with area proportional to total influence."""
        from utils.viz.screen import plot_communication_screen
        return plot_communication_screen(summary, **kwargs)

    def receiver_response(self, receiver, *, genes, components=None, adata=None,
                          checkpoint_path=None, **kwargs):
        """Point level: exact sender/module/gene decomposition at one receiver.

        Pass ``components=analysis.response_components()`` to reuse components
        in memory. Otherwise they are read from the selected fitted checkpoint.
        All modules remain in the decomposition, even if the plot highlights few.
        """
        from utils.tools.hierarchy import receiver_response
        target = self.adata if adata is None else adata
        if components is None:
            components = self.response_components(adata=target, checkpoint_path=checkpoint_path)
        return receiver_response(target, receiver, genes=genes,
                                 prefix=self.config.eval.velocity_key, **components, **kwargs)

    @staticmethod
    def plot_receiver_response(result, **kwargs):
        """Plot the point-level sender routes and signed gene-response ledger."""
        from utils.viz.hierarchy import plot_receiver_response
        return plot_receiver_response(result, **kwargs)

    def tissue_communication(self, *, adata=None, **kwargs):
        """Tissue level: spatial input routes, receiver maps and source attribution."""
        from utils.tools.hierarchy import tissue_communication
        return tissue_communication(self.adata if adata is None else adata,
                                    prefix=self.config.eval.velocity_key, **kwargs)

    @staticmethod
    def plot_sender_catchment(result, **kwargs):
        """Plot sender catchment in tissue coordinates without spatial binning."""
        from utils.viz.hierarchy import plot_sender_catchment
        return plot_sender_catchment(result, **kwargs)

    def communication_statistics(self, module, *, sections=None, **kwargs):
        """Summary level: source -> section -> observed receiver-state input routing.

        Pass ``sections={section_id: fitted_adata, ...}`` for a multi-section
        summary, plus source labels, receiver masks and observed state values.
        Explicit donor IDs enable donor-balanced weights across technical sections.
        See :func:`utils.tools.statistics.communication_statistics`.
        """
        from utils.tools.statistics import communication_statistics
        target = {"section": self.adata} if sections is None else sections
        return communication_statistics(target, module, prefix=self.config.eval.velocity_key, **kwargs)

    @staticmethod
    def plot_communication_statistics(result, **kwargs):
        """Plot source/section/state Sankey flows with conserved influence widths."""
        from utils.viz.statistics import plot_communication_statistics
        return plot_communication_statistics(result, **kwargs)

    def evaluate_metrics(
        self,
        *,
        adata: Optional[ad.AnnData] = None,
        cluster_key: str = "clusters",
        cluster_edges: Sequence[tuple[str, str]] | None = None,
        tprior_key: str | None = None,
        dataset_name: str | None = None,
        seed: int | None = None,
        n_gene_plots: int = 0,
        n_jobs: int | None = None,
    ) -> pd.DataFrame:
        """
        RNA velocity benchmark metrics (TopoVelo ``VelocityEvaluator``).

        Pass ``seed`` for simulation benchmarks (jittered ``X_spatial_knn`` + ``spatial_graph``).
        """
        target = adata if adata is not None else self.adata
        key = self.config.eval.velocity_key
        vel_layer = f"{key}_velocity"
        time_key = f"{key}_time"
        if vel_layer not in target.layers:
            raise KeyError(f"Missing layers[{vel_layer!r}]")
        if time_key not in target.obs:
            raise KeyError(f"Missing obs[{time_key!r}]")
        if cluster_key not in target.obs:
            raise KeyError(f"Missing obs[{cluster_key!r}]")

        prepared = prepare_metrics_adata(target, key, seed=seed)
        if tprior_key is not None:
            prepared.obs["tprior"] = np.asarray(prepared.obs[tprior_key], dtype=np.float64)
        spatial_key = SPATIAL_KNN_KEY if seed is not None else "spatial"
        label = dataset_name or str(prepared.uns.get("dataset_name", key))
        evaluator = VelocityEvaluator(
            Path(self.config.eval.output_dir).parent / "perf",
            spatial_key=spatial_key,
            cluster_key=cluster_key,
        )
        evaluator.evaluate(
            prepared,
            label,
            label,
            ["CellVOCAL"],
            [key],
            list(cluster_edges or ()),
            spatial_velocity_graph=True,
            n_gene_plots=n_gene_plots,
            seed=seed,
            n_jobs=n_jobs,
        )
        evaluator.save(key)
        return evaluator.summary()

    def evaluate_ccc_recovery(
        self,
        *,
        adata: Optional[ad.AnnData] = None,
        pred_prefix: Optional[str] = None,
        gt_key: str = "ground_truth",
        output_dir: Optional[Union[str, Path]] = None,
        output_path: Optional[Union[str, Path]] = None,
        spatial_key: str = "spatial",
        spatial_graph_key: str | None = None,
        n_neighbors: int | None = None,
        max_distance: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate event, receiver-module, and sender-receiver CCC recovery.

        Requires ``adata.uns[gt_key]['events']`` (simulation ground truth) and
        ``adata.uns[{pred_prefix}_ccc]`` (CellVOCAL export).

        When ``output_dir`` or ``output_path`` is set, writes
        ``*_roc.csv`` and ``*_topk.csv`` with the data used by the report.
        """
        target = adata if adata is not None else self.adata
        prefix = pred_prefix if pred_prefix is not None else self.config.eval.velocity_key
        graph_key = (
            spatial_graph_key
            if spatial_graph_key is not None
            else self.config.graph.spatial_graph_key
        )
        n_nb = n_neighbors if n_neighbors is not None else self.config.graph.n_spatial_neighbors
        result = CCCRecoveryEvaluator.from_anndata(
            target,
            pred_prefix=prefix,
            gt_key=gt_key,
            spatial_key=spatial_key,
            spatial_graph_key=graph_key,
            n_neighbors=n_nb,
            max_distance=max_distance,
        )
        save_target = output_path if output_path is not None else output_dir
        if save_target is not None:
            CCCRecoveryEvaluator().save(result, save_target)
        return result.roc_table(), result.topk_table()

    def plot_velocity(
        self,
        *,
        adata: Optional[ad.AnnData] = None,
        output_dir: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
        cluster_key: str = "clusters",
        velocity_color_keys: Sequence[str] | None = None,
        spatial_fill: float | None = None,
        n_jobs: int = 8,
        grid_size: int = 17,
        min_cells: int = 6,
        max_cells: int = 2200,
        seed: int = 2022,
        palette: Mapping[str, str] | None = None,
    ) -> Tuple[Path, ...]:
        """Plot time and grid-averaged velocity directions, as in Ext. Fig. 1a.

        Equal-length arrows show direction, not speed. Only background cells
        are subsampled; all finite cells contribute to the velocity grid.
        Pass a category-to-color ``palette`` to keep colors fixed across samples.
        """
        target = adata if adata is not None else self.adata
        key = tag if tag is not None else self.config.eval.velocity_key
        out = Path(
            output_dir if output_dir is not None else self.config.eval.output_dir / "figures"
        )
        return VelocityEmbeddingPlotter(
            out,
            tag=key,
            cluster_key=cluster_key,
            velocity_color_keys=velocity_color_keys,
            spatial_fill=spatial_fill,
            n_jobs=n_jobs,
            grid_size=grid_size,
            min_cells=min_cells,
            max_cells=max_cells,
            seed=seed,
            palette=palette,
        ).plot(target)

    def umap_transfer(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        adata: Optional[ad.AnnData] = None,
        output_dir: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
        n_neighbors: int = 30,
        n_pcs: int = 30,
        n_clusters: Optional[int] = None,
        dpi: int = 300,
    ) -> Path:
        """
        Plot Translator UMAP with Louvain clustering.

        Writes ``translator.png`` under
        ``output_dir`` (default: ``{eval.output_dir}/translator``).
        """
        key = tag if tag is not None else self.config.eval.velocity_key
        out = Path(
            output_dir
            if output_dir is not None
            else self.config.eval.output_dir / "translator"
        )
        transfer = TranslatorUmapTransfer(
            out,
            prefix=key,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            n_clusters=n_clusters,
            dpi=dpi,
        )
        if path is not None:
            return transfer.run_path(path)
        target = adata if adata is not None else self.adata
        return transfer.plot(target)

    def translator_report(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        adata: Optional[ad.AnnData] = None,
        output_dir: Optional[Union[str, Path]] = None,
        tag: Optional[str] = None,
        n_neighbors: int = 30,
        n_pcs: int = 30,
        n_clusters: Optional[int] = None,
        dpi: int = 300,
        plot_max_events: int | None = None,
        metric_max_events: int | None = None,
        gt_key: str = "ground_truth",
        label_columns: Sequence[str] | None = None,
        knn_k: int = 15,
        point_size: float | None = None,
        point_alpha: float | None = None,
    ) -> dict[str, Any]:
        """
        Plot event-level Translator UMAP diagnostics and partition metrics.

        Writes ``events_by_*.png``, ``events_metadata.csv``, and
        ``event_partition_metrics.csv`` under ``output_dir``
        (default: ``{eval.output_dir}/translator``).
        """
        key = tag if tag is not None else self.config.eval.velocity_key
        out = Path(
            output_dir
            if output_dir is not None
            else self.config.eval.output_dir / "translator"
        )
        transfer = TranslatorUmapTransfer(
            out,
            prefix=key,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            n_clusters=n_clusters,
            dpi=dpi,
        )
        if path is not None:
            return transfer.run_report_path(
                path,
                plot_max_events=plot_max_events,
                metric_max_events=metric_max_events,
                gt_key=gt_key,
                label_columns=label_columns,
                knn_k=knn_k,
                point_size=point_size,
                point_alpha=point_alpha,
            )
        target = adata if adata is not None else self.adata
        return transfer.plot_report(
            target,
            plot_max_events=plot_max_events,
            metric_max_events=metric_max_events,
            gt_key=gt_key,
            label_columns=label_columns,
            knn_k=knn_k,
            point_size=point_size,
            point_alpha=point_alpha,
        )

    def plot_lr_spatial(
        self,
        *,
        adata: Optional[ad.AnnData] = None,
        output_dir: Optional[Union[str, Path]] = None,
        lr_pairs: Optional[list[str]] = None,
        top_fraction: float | None = None,
        figsize: Optional[tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
    ) -> list[Path]:
        """
        Plot per–ligand–receptor spatial communication maps from CCC inferred signalling influence.

        For each LR pair, keeps the top ``top_fraction`` of positive spatial edges
        by inferred signalling influence score. Figures are written as SVG under ``output_dir``.
        """
        target = adata if adata is not None else self.adata
        out = Path(
            output_dir
            if output_dir is not None
            else self.config.eval.output_dir / "lr_spatial"
        )
        frac = (
            self.config.eval.lr_spatial_top_fraction
            if top_fraction is None
            else top_fraction
        )
        selected = parse_lr_pairs(lr_pairs) if lr_pairs is not None else None
        plotter = LRSpatialPlotter(out, prefix=self.config.eval.velocity_key)
        return plotter.plot(
            target,
            ligand_receptor_pairs=selected,
            top_fraction=frac,
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
        )

    def plot_lr_spatial_by_region(
        self,
        *,
        adata: Optional[ad.AnnData] = None,
        output_dir: Optional[Union[str, Path]] = None,
        lr_pairs: Optional[list[str]] = None,
        top_fraction: float | None = None,
        figsize: Optional[tuple[int, int]] = None,
        max_outgoing_edges_per_node: int = 3,
        max_incoming_edges_per_node: int = 3,
        region_key: str = "clusters",
    ) -> list[Path]:
        """
        Same as :meth:`plot_lr_spatial`, with cell polygons colored by region.

        Requires ``adata.obs[region_key]`` (e.g. ``clusters`` or ``domain``).
        """
        target = adata if adata is not None else self.adata
        out = Path(
            output_dir
            if output_dir is not None
            else self.config.eval.output_dir / "lr_spatial_region"
        )
        frac = (
            self.config.eval.lr_spatial_top_fraction
            if top_fraction is None
            else top_fraction
        )
        selected = parse_lr_pairs(lr_pairs) if lr_pairs is not None else None
        plotter = LRSpatialPlotter(out, prefix=self.config.eval.velocity_key)
        return plotter.plot_by_region(
            target,
            ligand_receptor_pairs=selected,
            top_fraction=frac,
            figsize=figsize,
            max_outgoing_edges_per_node=max_outgoing_edges_per_node,
            max_incoming_edges_per_node=max_incoming_edges_per_node,
            region_key=region_key,
        )
