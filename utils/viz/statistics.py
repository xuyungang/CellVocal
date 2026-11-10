"""Source -> section -> receiver-state Sankey summaries."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
import numpy as np

from .communication_style import style_communication_figure


def _ribbon(ax, x0, x1, y0, y1, height, color, alpha):
    if height <= 0:
        return
    dx = x1 - x0
    vertices = [(x0, y0), (x0+.4*dx, y0), (x1-.4*dx, y1), (x1, y1),
                (x1, y1+height), (x1-.4*dx, y1+height), (x0+.4*dx, y0+height),
                (x0, y0+height), (x0, y0)]
    codes = [Path.MOVETO, *[Path.CURVE4]*3, Path.LINETO, *[Path.CURVE4]*3, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(vertices, codes), facecolor=color, edgecolor="none",
                          alpha=alpha, rasterized=True))


def plot_communication_statistics(
    result, *, ax=None, source_labels=None, section_labels=None, state_labels=None,
    source_colors=None, section_colors=None, state_colors=None,
    state_title="Receiver state", title=None, show_width_label=True, figsize=(6.8, 2.4),
):
    """Plot the conserved source/section/state tensor and return ``(figure, ax)``.

    All links share a common width scale. Left links sum over receiver states;
    right links sum over source classes. These are two marginals of the same
    allocation, not independently normalized or directed cell-state transitions.
    Label/color mappings use the IDs stored in result["metadata"].
    """
    meta = result["metadata"]
    orders = [meta["source_order"], meta["section_order"], meta["state_order"]]
    values = np.asarray(result["tensor"], float)
    if values.shape != tuple(map(len, orders)) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("tensor must be finite, nonnegative, and match the source/section/state axes")
    if any(not order for order in orders):
        raise ValueError("All three routing axes must contain at least one node")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure
    palettes = [
        ("#7EA4A7", "#AD7790", "#A6AFB4", "#83A8C0", "#D0AE76", "#91B2A0", "#A695B6"),
        ("#708EAC", "#478F8C", "#C49B61", "#8970AA", "#B64B5D"),
        ("#E9DCC2", "#C49B61", "#95703A", "#749EAB", "#86649A"),
    ]
    mappings = [source_labels or {}, section_labels or {}, state_labels or {}]
    color_maps = [source_colors or {}, section_colors or {}, state_colors or {}]
    colors = [[mapping.get(name, palette[i % len(palette)]) for i, name in enumerate(order)]
              for order, mapping, palette in zip(orders, color_maps, palettes)]
    left, right = values.sum(axis=2), values.sum(axis=0)
    totals = [left.sum(axis=1), left.sum(axis=0), right.sum(axis=0)]
    # Equally spaced node centres match the compact reference panel; one common
    # scale ensures width is comparable across every link and every column.
    centres = [np.linspace(.82, .16, len(order)) if len(order) > 1 else np.array([.49])
               for order in orders]
    maximum = max(float(v.max()) for v in totals)
    spacing = min(.66 / (len(order)-1) if len(order) > 1 else .66 for order in orders)
    scale = min(.32, spacing*.85) / maximum if maximum > 0 else 0.
    cursor = [y+v*scale/2 for y, v in zip(centres, totals)]
    middle_out = cursor[1].copy()
    for i in range(len(orders[0])):
        for j in range(len(orders[1])):
            height = left[i, j]*scale
            cursor[0][i] -= height
            cursor[1][j] -= height
            _ribbon(ax, .16, .45, cursor[0][i], cursor[1][j], height, colors[0][i], .68)
    for j in range(len(orders[1])):
        for k in range(len(orders[2])):
            height = right[j, k]*scale
            middle_out[j] -= height
            cursor[2][k] -= height
            _ribbon(ax, .51, .84, middle_out[j], cursor[2][k], height, colors[1][j], .60)
    for col, (x, width) in enumerate(((.135, .025), (.45, .06), (.84, .025))):
        for i, name in enumerate(orders[col]):
            height = totals[col][i]*scale
            ax.add_patch(Rectangle((x, centres[col][i]-height/2), width, height,
                                   facecolor=colors[col][i], edgecolor="none"))
            label = mappings[col].get(name, name)
            if col == 0:
                ax.text(.115, centres[col][i], label, ha="right", va="center", fontsize=6)
            elif col == 1:
                ax.text(.48, centres[col][i]+height/2+.025, label, ha="center", va="bottom", fontsize=6)
            else:
                ax.text(.89, centres[col][i], label, va="center", fontsize=6)
    for x, heading in ((.08, "Source"), (.48, "Section"), (.95, state_title)):
        ax.text(x, 1.04, heading, ha="center", fontsize=6)
    if show_width_label:
        ax.text(.5, -.055, f"Width: {meta['width_label'].lower()}",
                ha="center", va="top", fontsize=5)
    if title:
        ax.set_title(title, fontsize=6, loc="left", pad=12)
    if maximum == 0:
        ax.text(.48, .49, "No positive input", ha="center", fontsize=6, color="#64717A")
    ax.set(xlim=(-.06, 1.12), ylim=(-.09, 1.10))
    ax.axis("off")
    style_communication_figure(fig)
    return fig, ax
