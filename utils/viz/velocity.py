from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence, Tuple

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from pandas.api.types import is_numeric_dtype

from model.train.data import DEFAULT_SPATIAL_KEY
from utils.metric.velocity import embed_spatial_velocity
from .spatial_velocity import (
    CELL_TYPE_PALETTE,
    draw_velocity_arrows,
    grid_vectors,
    map_limits,
    normalize_time,
    spatial_sample,
    style_spatial_axis,
)

_RC: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "axes.grid": False,
    "axes.spines.left": False,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.spines.bottom": False,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
}


class VelocityEmbeddingPlotter:
    """Spatial time and velocity maps using the Ext. Fig. 1a grid logic.

    Average all finite cell velocities in a 17-by-17 grid, retaining bins with
    at least six cells. Equal-length arrows show direction, not speed. Sample
    at most 2,200 background cells, and min-max normalize numeric colors.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        tag: str = "velocity",
        cluster_key: str = "clusters",
        spatial_key: str = DEFAULT_SPATIAL_KEY,
        dpi: int = 300,
        n_jobs: int = 8,
        latent_time_kwargs: dict[str, Any] | None = None,
        velocity_color_keys: Sequence[str] | None = None,
        spatial_fill: float | None = None,
        grid_size: int = 17,
        min_cells: int = 6,
        max_cells: int = 2200,
        seed: int = 2022,
        palette: Mapping[str, str] | None = None,
    ) -> None:
        if grid_size < 2 or min_cells < 1:
            raise ValueError("grid_size must be at least 2 and min_cells positive.")
        self.output_dir = Path(output_dir)
        self.tag = tag
        self.cluster_key = cluster_key
        self.spatial_key = spatial_key
        self.dpi = dpi
        self.n_jobs = n_jobs
        self._lt_kw = latent_time_kwargs or {}
        self.velocity_color_keys = (
            tuple(velocity_color_keys) if velocity_color_keys is not None else None
        )
        # Retain the public density control as a relative marker-area scale.
        self.spatial_fill = 0.55 if spatial_fill is None else float(spatial_fill)
        self.grid_size = grid_size
        self.min_cells = min_cells
        self.max_cells = max_cells
        self.seed = seed
        self.palette = dict(palette or {})

    @property
    def _vkey(self) -> str:
        return f"{self.tag}_velocity"

    def _spatial_xy(self, adata: ad.AnnData) -> np.ndarray:
        xy = np.asarray(adata.obsm[self.spatial_key], dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"obsm[{self.spatial_key!r}] must have at least two columns.")
        return xy[:, :2]

    def _velocity_color_keys(self, adata: ad.AnnData) -> tuple[str, ...]:
        if self.velocity_color_keys is None:
            candidates = (f"{self.tag}_time", self.cluster_key)
            keys: list[str] = []
            for key in candidates:
                if key in adata.obs and key not in keys:
                    keys.append(key)
            if keys:
                return tuple(keys)
            raise KeyError(
                "Velocity maps require at least one color key in obs. "
                f"Tried {list(candidates)!r}; available obs columns: {list(adata.obs.columns)!r}"
            )

        missing = [key for key in self.velocity_color_keys if key not in adata.obs]
        if missing:
            raise KeyError(
                "Missing requested velocity color keys in obs: "
                + ", ".join(repr(key) for key in missing)
                + f". Available obs columns: {list(adata.obs.columns)!r}"
            )
        keys = []
        for key in self.velocity_color_keys:
            if key not in keys:
                keys.append(key)
        return tuple(keys)

    @staticmethod
    def _filename_token(key: str) -> str:
        token = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(key)
        )
        return token.strip("_") or "obs"

    def _plot_map(
        self,
        adata: ad.AnnData,
        *,
        color_key: str,
        output_path: Path,
        title: str,
        velocity: np.ndarray | None = None,
        cbar_label: str = "Normalized time",
    ) -> Path:
        if color_key not in adata.obs:
            raise KeyError(f"Missing obs[{color_key!r}]")
        xy = self._spatial_xy(adata)
        values = adata.obs[color_key]
        numeric = is_numeric_dtype(values)
        colors = (values.to_numpy(dtype=float, na_value=np.nan) if numeric
                  else values.astype(str).to_numpy())
        finite = np.isfinite(xy).all(axis=1)
        if velocity is not None:
            velocity = np.asarray(velocity, dtype=np.float64)
            if velocity.ndim != 2 or velocity.shape[0] != len(xy) or velocity.shape[1] < 2:
                raise ValueError("Spatial velocity must have one row per cell and two columns.")
            velocity = velocity[:, :2]
            finite &= np.isfinite(velocity).all(axis=1)
        if numeric:
            finite &= np.isfinite(colors)
        if not finite.any():
            raise ValueError("No finite cells are available for the spatial map.")
        xy, colors = xy[finite], colors[finite]
        if numeric:
            colors = normalize_time(colors)
        display = spatial_sample(xy, self.max_cells, seed=self.seed)
        size_scale = (self.spatial_fill / 0.55) ** 2

        with plt.rc_context(_RC):
            fig, ax = plt.subplots(figsize=(3.4, 3.4), facecolor="white")
            # Reserve the same map area for time-colored and categorical panels.
            fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.20)
            if numeric:
                points = ax.scatter(
                    xy[display, 0], xy[display, 1], c=colors[display],
                    s=1.4 * size_scale, cmap="cividis", vmin=0.0, vmax=1.0,
                    alpha=0.24 if velocity is not None else 0.82,
                    linewidths=0, rasterized=True, zorder=1,
                )
                cax = fig.add_axes([0.23, 0.105, 0.54, 0.018])
                bar = fig.colorbar(points, cax=cax, orientation="horizontal")
                bar.ax.xaxis.set_label_position("top")
                bar.set_label(cbar_label, labelpad=2)
                bar.set_ticks((0.0, 0.5, 1.0))
                bar.outline.set_linewidth(0.45)
            else:
                # A supplied palette keeps category colors fixed across datasets.
                categories = list(dict.fromkeys(colors))
                for i, category in enumerate(categories):
                    members = display[colors[display] == category]
                    ax.scatter(
                        xy[members, 0], xy[members, 1],
                        s=2.0 * size_scale,
                        color=self.palette.get(
                            category, CELL_TYPE_PALETTE[i % len(CELL_TYPE_PALETTE)],
                        ),
                        alpha=0.82, linewidths=0, rasterized=True, zorder=1,
                        label=str(category).replace("_region_", " R").replace("_", " "),
                    )
                fig.legend(
                    *ax.get_legend_handles_labels(), loc="upper center",
                    bbox_to_anchor=(0.5, 0.17), ncol=min(3, len(categories)),
                    columnspacing=0.7, handletextpad=0.25, markerscale=2.0,
                )
            if velocity is not None:
                grid = grid_vectors(
                    xy, velocity[finite], colors if numeric else np.zeros(len(xy)),
                    grid_size=self.grid_size, min_cells=self.min_cells,
                    velocity_prefix="velocity", time_column="color_normalized",
                )
                draw_velocity_arrows(ax, xy, grid, ("velocity_vx", "velocity_vy"))
            style_spatial_axis(ax, map_limits(xy))
            ax.set_title(title, pad=5)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=self.dpi, facecolor="white", transparent=False)
            plt.close(fig)
        print(f"[plot] {output_path}")
        return output_path

    def plot_latent_time(self, adata: ad.AnnData) -> Path:
        return self._plot_map(
            adata, color_key=f"{self.tag}_time",
            output_path=self.output_dir / "latent_time_spatial.png",
            title="Inferred time",
            cbar_label=str(self._lt_kw.get("cbar_label", "Normalized time")),
        )

    def plot_spatial_velocity(
        self,
        adata: ad.AnnData,
        velocity_key: str,
        output_path: str | Path,
        *,
        color_key: str | None = None,
        title: str | None = None,
    ) -> Path:
        """Plot a precomputed 2D velocity field as occupied-bin direction arrows."""
        if velocity_key not in adata.obsm:
            raise KeyError(f"Missing obsm[{velocity_key!r}]")
        return self._plot_map(
            adata, color_key=color_key or self.cluster_key,
            velocity=adata.obsm[velocity_key], output_path=Path(output_path),
            title=title or velocity_key.replace("_", " "),
        )

    def plot_spatial_velocity_stream(
        self,
        adata: ad.AnnData,
        velocity_key: str,
        output_path: str | Path,
        *,
        color_key: str | None = None,
        title: str | None = None,
    ) -> Path:
        """Compatibility alias; spatial velocity now uses grid arrows."""
        return self.plot_spatial_velocity(
            adata, velocity_key, output_path, color_key=color_key, title=title,
        )

    def plot_velocity_maps(self, adata: ad.AnnData) -> Tuple[Path, ...]:
        if self._vkey not in adata.layers:
            raise KeyError(f"RNA velocity plots require layer {self._vkey!r}")
        color_keys = self._velocity_color_keys(adata)
        # As in Ext. Fig. 1a, embed the full RNA-velocity layer before binning.
        # Work on a copy so visualization never alters the fitted AnnData.
        viz = adata.copy()
        viz.obsm["X_spatial"] = self._spatial_xy(viz)
        velocity = embed_spatial_velocity(
            viz, self.tag, n_jobs=self.n_jobs,
            spatial_key=self.spatial_key, inplace=True,
        )
        paths = []
        for key in color_keys:
            suffix = self._filename_token(key)
            label = "pseudotime" if key == f"{self.tag}_time" else key.replace("_", " ")
            paths.append(self._plot_map(
                viz, color_key=key, velocity=velocity,
                title=f"Total RNA velocity ({label})",
                output_path=self.output_dir / f"spatial_total_velocity_quiver_by_{suffix}.png",
            ))
        return tuple(paths)

    def plot_velocity_streams(self, adata: ad.AnnData) -> Tuple[Path, ...]:
        """Compatibility alias; returns the grid-arrow maps."""
        return self.plot_velocity_maps(adata)

    def plot(self, adata: ad.AnnData) -> Tuple[Path, ...]:
        """Return a time map followed by velocity maps for requested obs keys."""
        return self.plot_latent_time(adata), *self.plot_velocity_maps(adata)
