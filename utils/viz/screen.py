"""Bubble-screen visualisation with caller-defined RNA-response direction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .communication_style import style_communication_figure


def plot_communication_screen(
    summary: pd.DataFrame,
    *,
    ax=None,
    highlights: Mapping[str, str] | Sequence[str] | None = None,
    module_labels: Mapping[str, str] | None = None,
    outcome_label: str = "Outcome",
    outcome_units: str | None = None,
    contrast_scale: float = 1.0,
    max_marker_area: float = 85.0,
    background_color: str = "#BCC3C8",
    fontsize: float = 6.0,
    figsize: tuple[float, float] = (3.4, 2.5),
    size_legend: bool = True,
    rasterized: bool = True,
):
    """Plot equal-section correlation versus high-input/background contrast.

    ``summary`` is the second return value of
    :func:`utils.tools.screen.screen_communication_modules`.
    Bubble *area*, not radius, is proportional to summed inferred signalling influence.
    ``highlights`` supplies selected modules or a module-to-colour mapping;
    the function never selects modules based on their observed effects.
    ``contrast_scale`` is an explicit display conversion only (e.g. 1000
    when the accompanying unit label is ``"0.001 input SD"``).

    Returns ``(figure, axes)`` without writing files or changing global
    plotting settings. Non-finite statistics are omitted with a warning;
    the original source table retains those candidates and their statuses.
    """
    required = {"module", "mean_rho", "mean_high_delta", "total_influence"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary is missing columns: {sorted(required - set(summary.columns))}")
    if summary.module.duplicated().any():
        raise ValueError("summary must contain exactly one row per module")
    if not np.isfinite(contrast_scale) or contrast_scale <= 0:
        raise ValueError("contrast_scale must be a positive finite number")
    if not np.isfinite(max_marker_area) or max_marker_area <= 0:
        raise ValueError("max_marker_area must be a positive finite number")
    total_influence = summary.total_influence.to_numpy(dtype=float)
    if (total_influence[np.isfinite(total_influence)] < 0).any():
        raise ValueError("total_influence must be non-negative")
    if highlights is None:
        colours = {}
    elif isinstance(highlights, Mapping):
        colours = dict(highlights)
    else:
        if isinstance(highlights, str):
            highlights = [highlights]
        palette = ("#B54B50", "#228D86", "#C49B61", "#8B659E")
        colours = {name: palette[i % len(palette)] for i, name in enumerate(highlights)}
    absent = set(colours) - set(summary.module)
    if absent:
        raise ValueError(f"Highlighted modules are absent from summary: {sorted(absent)}")
    labels = {} if module_labels is None else module_labels
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    fig = ax.figure
    finite = np.isfinite(
        summary[["mean_rho", "mean_high_delta", "total_influence"]].to_numpy(dtype=float)
    ).all(axis=1)
    if not finite.all():
        warnings.warn(
            f"{int((~finite).sum())} module(s) have undefined screen statistics and are not plotted",
            RuntimeWarning,
            stacklevel=2,
        )
    table = summary.loc[finite].copy()
    maximum = float(np.nanmax(total_influence)) if np.isfinite(total_influence).any() else 0.0
    area_scale = max_marker_area / maximum if maximum > 0 else 0.0
    table["marker_area"] = table.total_influence * area_scale
    selected = table.module.isin(colours)
    for use in (~selected, selected):
        subset = table.loc[use]
        if subset.empty:
            continue
        ax.scatter(
            subset.mean_rho,
            subset.mean_high_delta * contrast_scale,
            s=subset.marker_area,
            c=[colours.get(module, background_color) for module in subset.module],
            edgecolors="white",
            linewidths=0.35,
            alpha=1.0 if use.equals(selected) else 0.8,
            rasterized=rasterized,
            zorder=3 if use.equals(selected) else 2,
        )
    for row in table.loc[selected].itertuples():
        ax.annotate(
            labels.get(row.module, str(row.module).replace("->", "–")),
            (row.mean_rho, row.mean_high_delta * contrast_scale),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=fontsize,
            color=colours[row.module],
            ha="left",
            va="bottom",
        )
    ax.axhline(0, color="#DFE3E6", lw=0.5, zorder=0)
    ax.axvline(0, color="#DFE3E6", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(f"Input–{outcome_label.lower()} coupling (Spearman ρ)", fontsize=fontsize)
    unit_label = f"\n({outcome_units})" if outcome_units else ""
    ax.set_ylabel(f"High-input − background {outcome_label.lower()}{unit_label}", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize, length=2, width=0.6)
    ax.margins(x=0.18, y=0.22)
    if size_legend and maximum > 0 and not table.empty:
        levels = np.unique(np.round(np.array([0.25, 0.75]) * maximum, 2))
        levels = levels[levels > 0]
        handles = [
            ax.scatter([], [], s=value * area_scale, c=background_color, edgecolors="white", lw=0.35)
            for value in levels
        ]
        ax.legend(
            handles,
            [f"{value:,.3g}" for value in levels],
            title="Σ inferred signalling influence",
            loc="best",
            frameon=False,
            fontsize=fontsize,
            title_fontsize=fontsize,
            ncol=max(1, len(levels)),
            borderpad=0.2,
            handletextpad=0.4,
            columnspacing=0.7,
        )
    style_communication_figure(fig)
    return fig, ax
