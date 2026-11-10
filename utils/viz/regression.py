"""Compact display of communication event-rate regression contrasts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from .communication_style import style_communication_figure


def plot_communication_regression(
    results: pd.DataFrame,
    *,
    ax=None,
    group_order: Sequence[str] | None = None,
    contrast_order: Sequence[str] | None = None,
    fdr_threshold: float = 0.05,
    cmap: str = "RdBu_r",
    vmax: float | None = None,
    colorbar: bool = True,
):
    """Plot log2 event-rate ratios, with FDR dots and unavailable-inference x's.

    Returns ``(figure, axes)`` without saving or showing.  Labels and markers
    are vector artists.  When saving, use ``svg.fonttype='none'`` or
    ``pdf.fonttype=42`` in a matplotlib rc context to preserve editable text.
    Empty or duplicated group/contrast entries are rejected rather than
    silently aggregated.
    """
    import matplotlib.pyplot as plt

    required = {"group_label", "contrast", "log2_rate_ratio", "fdr"}
    if not isinstance(results, pd.DataFrame) or not required.issubset(results):
        raise ValueError(f"results must contain {sorted(required)}.")
    if results.empty or results.duplicated(["group_label", "contrast"]).any():
        raise ValueError("results must have unique, nonempty group/contrast rows.")
    if not np.isfinite(fdr_threshold) or not 0 < fdr_threshold <= 1:
        raise ValueError("fdr_threshold must lie in (0, 1].")
    groups = list(pd.unique(results.group_label)) if group_order is None else list(group_order)
    contrasts = list(pd.unique(results.contrast)) if contrast_order is None else list(contrast_order)
    if len(set(groups)) != len(groups) or len(set(contrasts)) != len(contrasts):
        raise ValueError("Display orders cannot contain duplicates.")
    if not groups or not contrasts or not set(groups).issubset(results.group_label) or not set(contrasts).issubset(results.contrast):
        raise ValueError("Display orders must contain existing groups and contrasts.")
    matrix = results.pivot(index="group_label", columns="contrast", values="log2_rate_ratio").reindex(
        index=groups, columns=contrasts,
    )
    fdr = results.pivot(index="group_label", columns="contrast", values="fdr").reindex(
        index=groups, columns=contrasts,
    ).to_numpy(float)
    values = matrix.to_numpy(float)
    finite = np.abs(values[np.isfinite(values)])
    limit = float(max(finite.max(), 1e-6)) if vmax is None and len(finite) else (1.0 if vmax is None else float(vmax))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("vmax must be finite and positive.")
    rc = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5,
        "ytick.labelsize": 5, "pdf.fonttype": 42, "svg.fonttype": "none",
    }
    with plt.rc_context(rc):
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(2.4, .62 * len(contrasts)), max(1.5, .18 * len(groups))))
        else:
            fig = ax.figure
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        sig_y, sig_x = np.where(np.isfinite(fdr) & (fdr < fdr_threshold))
        ax.scatter(sig_x, sig_y, marker="o", s=4, c="#252525", linewidths=0, label=f"FDR < {fdr_threshold:g}")
        unavailable_y, unavailable_x = np.where(~np.isfinite(fdr))
        ax.scatter(unavailable_x, unavailable_y, marker="x", s=9, c="#555555", linewidths=.5, label="Inference unavailable")
        label_map = (
            results.drop_duplicates("contrast").set_index("contrast").contrast_label.to_dict()
            if "contrast_label" in results else {}
        )
        ax.set_xticks(np.arange(len(contrasts)), [label_map.get(value, str(value)) for value in contrasts], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(groups)), groups)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if colorbar:
            bar = fig.colorbar(image, ax=ax, fraction=.05, pad=.04)
            bar.set_label("Event-rate ratio (log2)")
            bar.outline.set_visible(False)
            bar.ax.tick_params(length=2, width=.4)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1), fontsize=5, ncols=2, handletextpad=.3)
    style_communication_figure(fig)
    return fig, ax
