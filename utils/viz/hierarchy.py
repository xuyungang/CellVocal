"""Compact plots for receiver attribution and spatial sender catchments."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.transforms import Bbox
import numpy as np
from utils.viz.communication_style import style_communication_figure

_COLORS = ("#87609B", "#369A99", "#C39957", "#7090AE", "#B75A6D", "#6C806A")
_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
}


def _separate_module_labels(ax, labels, sender_labels, receiver_label):
    """Keep module text away from node circles at the actual rendered size."""
    renderer = ax.figure.canvas.get_renderer()
    obstacles = [text.get_window_extent(renderer).expanded(1.9, 1.9) for text in sender_labels]
    obstacles.append(receiver_label.get_window_extent(renderer).expanded(1.15, 4.5))
    offsets = [(0., 0.)]
    for distance in (.3, .5, .7):
        offsets.extend((dx * distance, dy * distance) for dx, dy in
                       ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)))
    for label in labels:
        origin = np.asarray(label.get_position())
        best, best_score = origin, float("inf")
        for dx, dy in offsets:
            candidate = origin + [dx, dy]
            label.set_position(candidate)
            box = label.get_window_extent(renderer).padded(2)
            overlap = 0.
            for obstacle in obstacles:
                intersection = Bbox.intersection(box, obstacle)
                if intersection is not None:
                    overlap += intersection.width * intersection.height
            inside = Bbox.intersection(box, ax.bbox)
            outside = box.width * box.height - (0 if inside is None else inside.width * inside.height)
            score = overlap + outside + .01 * (dx * dx + dy * dy)
            if score < best_score:
                best, best_score = candidate, score
            if overlap == 0 and outside <= 1e-8:
                break
        label.set_position(best)
        obstacles.append(label.get_window_extent(renderer).padded(2))


def plot_receiver_response(
    result: dict[str, Any],
    *,
    modules: Sequence[str] | None = None,
    max_modules: int = 3,
    max_senders: int = 8,
    figsize: tuple[float, float] = (6.2, 2.4),
) -> tuple[Any, np.ndarray]:
    """Draw sender routes and an exact signed module-to-gene ledger.

    Modules not selected for display are combined into ``Other modules``.
    The black ticks are the sum across *all* modules. The route diagram limits
    source nodes by positive inferred signalling influence; this does not truncate the ledger.
    """
    table = result["module_response"]
    events = result["events"]
    names = table.module_name.drop_duplicates().tolist()
    if max_modules < 1 or max_senders < 1:
        raise ValueError("max_modules and max_senders must be positive.")
    if modules is None:
        ranked = table.drop_duplicates("module_name").sort_values(
            "module_activation_prob", ascending=False, kind="stable",
        )
        chosen = ranked.loc[ranked.module_activation_prob.gt(0), "module_name"].head(max_modules).tolist()
    else:
        chosen = list(modules)
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(names):
        raise ValueError("Displayed modules must be unique names present in the decomposition.")
    color = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(chosen)}
    with plt.rc_context(_RC):
        fig, axes = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.15]}, layout="constrained")
        ax, ga = axes
        shown = events.loc[events.module_name.isin(chosen) & events.event_influence.gt(0)].copy()
        senders = shown.groupby("sender", sort=True).event_influence.sum().nlargest(max_senders).index.to_numpy(int)
        angles = np.linspace(np.pi * .6, np.pi * 2.4, len(senders), endpoint=False)
        positions = {s: np.array([np.cos(t), np.sin(t)]) * 1.7 for s, t in zip(senders, angles)}
        port_angles = np.linspace(5 * np.pi / 6, 5 * np.pi / 6 - 2 * np.pi,
                                  len(chosen), endpoint=False)
        ports = {name: .9 * np.array([np.cos(angle), np.sin(angle)])
                 for name, angle in zip(chosen, port_angles)}
        maximum = max(float(shown.event_influence.max()), 1e-12) if len(shown) else 1.
        for row in shown.loc[shown.sender.isin(senders)].itertuples():
            ax.add_patch(FancyArrowPatch(
                positions[row.sender], ports[row.module_name], arrowstyle="-|>",
                connectionstyle="arc3,rad=.08", mutation_scale=4, shrinkA=5, shrinkB=3,
                color=color[row.module_name], lw=.3 + 1.2 * row.event_influence / maximum, alpha=.8,
            ))
        sender_labels = []
        module_labels = []
        for index, sender in enumerate(senders):
            ax.scatter(*positions[sender], s=100, facecolors="white", edgecolors="#69818A", linewidths=.6, zorder=4)
            sender_labels.append(ax.text(*positions[sender], f"S{index + 1}",
                                        ha="center", va="center", fontsize=5, zorder=5))
        ax.scatter([0], [0], s=600, facecolors="white", edgecolors="#51636E", linewidths=.6, zorder=3)
        receiver_label = ax.text(0, -.05, "Receiver", ha="center", va="center", zorder=5)
        for index, name in enumerate(chosen):
            ax.add_patch(FancyArrowPatch(
                ports[name], (0., 0.), arrowstyle="-|>", mutation_scale=4,
                shrinkA=3, shrinkB=11, color=color[name], lw=.7, zorder=4,
            ))
            ax.scatter(*ports[name], s=25, color=color[name], edgecolors="white", linewidths=.5, zorder=5)
            label = ports[name] * 1.42
            module_labels.append(ax.text(
                *label, name.replace("->", "\n"),
                ha="center", va="center", color=color[name], fontsize=5, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": .2},
            ))
        ax.set(xlim=(-2.1, 2.1), ylim=(-2.0, 2.1), aspect="equal", title="Sender-to-receiver routes")
        ax.axis("off")
        genes = result["gene_summary"].gene.tolist()
        omitted = [name for name in names if name not in chosen]
        legend = [Patch(facecolor=color[name], label=name.replace("->", "–")) for name in chosen]
        if omitted:
            legend.append(Patch(facecolor="#C0C7CA", label=f"Other {len(omitted)} modules"))
        for j, gene in enumerate(genes):
            part = table.loc[table.gene.eq(gene)].set_index("module_name").contribution
            values = [(part.loc[name], color[name]) for name in chosen]
            if omitted:
                values.append((part.loc[omitted].sum(), "#C0C7CA"))
            positive, negative = 0., 0.
            for value, shade in values:
                ga.barh(j, value, left=positive if value >= 0 else negative, height=.55, color=shade)
                if value >= 0:
                    positive += value
                else:
                    negative += value
            ga.plot([part.sum()] * 2, [j - .36, j + .36], color="#26333B", lw=.9)
        ga.axvline(0, color="#ABB6BD", lw=.6)
        ga.set(yticks=np.arange(len(genes)), yticklabels=genes, ylim=(len(genes) - .4, -.6),
               xlabel=f"Communication-associated velocity ({result['metadata']['units']})",
               title="Gene response at receiver")
        ga.spines[["top", "right", "left"]].set_visible(False)
        ga.tick_params(axis="y", length=0)
        legend.append(Line2D([0], [0], color="#26333B", marker="|", linestyle="none", label="All inputs"))
        ga.legend(handles=legend, frameon=False, fontsize=5, loc="upper center",
                  bbox_to_anchor=(.5, -.23), ncol=2)
        style_communication_figure(fig)
        fig.canvas.draw()
        _separate_module_labels(ax, module_labels, sender_labels, receiver_label)
    return fig, axes


def plot_sender_catchment(
    result: dict[str, Any],
    *,
    max_receivers: int = 6,
    max_routes_per_receiver: int = 2,
    focus_receiver: int | None = None,
    radius: float | None = None,
    invert_y: bool = True,
    figsize: tuple[float, float] = (4.0, 3.3),
) -> tuple[Any, Any]:
    """Plot tissue, receiver state, signed velocity rings and attributed routes.

    Representatives are selected by spatial separation, starting at the highest
    input receiver (or ``focus_receiver``). No velocity value enters selection.
    The strongest route for each module precedes additional routes when limiting
    density. All routes remain available in the analytical result. ``radius``
    optionally restricts the displayed receiver neighbourhood around an explicit
    ``focus_receiver`` in the native spatial coordinate units; senders outside
    that neighbourhood remain eligible and visible.
    """
    if max_receivers < 1 or max_routes_per_receiver < 1:
        raise ValueError("Route display limits must be positive.")
    sites, routes = result["sites"], result["routes"]
    mask = sites.is_receiver.to_numpy(bool).copy()
    if not mask.any():
        raise ValueError("No receiver sites were selected.")
    xy = sites[["x", "y"]].to_numpy(float)
    if radius is not None:
        if focus_receiver is None or not isinstance(focus_receiver, (int, np.integer)):
            raise ValueError("A spatial radius requires an integer focus_receiver.")
        if not 0 <= focus_receiver < len(sites) or not mask[focus_receiver]:
            raise ValueError("focus_receiver must identify a selected receiver.")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive.")
        mask &= np.linalg.norm(xy - xy[focus_receiver], axis=1) <= radius
        routes = routes.loc[mask[routes.receiver.to_numpy(int)]]
    eligible = np.sort(routes.receiver.unique()).astype(int)
    chosen = []
    if focus_receiver is not None:
        if focus_receiver not in eligible:
            raise ValueError("focus_receiver must be an input-positive selected receiver.")
        chosen.append(int(focus_receiver))
    elif len(eligible):
        chosen.append(int(sites.loc[eligible, "incoming_influence"].idxmax()))
    while chosen and len(chosen) < min(max_receivers, len(eligible)):
        distance = np.linalg.norm(xy[eligible, None] - xy[np.asarray(chosen)][None], axis=2).min(axis=1)
        distance[np.isin(eligible, chosen)] = -1
        chosen.append(int(eligible[distance.argmax()]))
    drawn = []
    quantity = result["metadata"]["route_quantity"]
    for receiver in chosen:
        part = routes.loc[routes.receiver.eq(receiver)].sort_values(quantity, ascending=False, kind="stable")
        first = part.drop_duplicates("module_name").index.tolist()
        remainder = [index for index in part.index if index not in first]
        drawn.extend((first + remainder)[:max_routes_per_receiver])
    shown = routes.loc[drawn]
    names = routes.module_name.drop_duplicates().tolist()
    colors = {name: _COLORS[i % len(_COLORS)] for i, name in enumerate(names)}
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        ax.scatter(*xy.T, s=2, color="#DDE2E3", linewidths=0, rasterized=True, zorder=0)
        if "observed_value" in sites:
            dots = ax.scatter(*xy[mask].T, s=13, c=sites.loc[mask, "observed_value"],
                              cmap="YlOrBr", edgecolors="white", linewidths=.2, rasterized=True, zorder=2)
            fig.colorbar(dots, ax=ax, orientation="horizontal", fraction=.045, pad=.04,
                         label=result["metadata"]["observed_label"])
        else:
            ax.scatter(*xy[mask].T, s=13, color="#B8D6D2", edgecolors="white",
                       linewidths=.2, rasterized=True, zorder=2)
        if "velocity_value" in sites:
            value = sites.loc[mask, "velocity_value"].to_numpy(float)
            width = .25 + np.abs(value) / max(float(np.abs(value).max()), 1e-12)
            ax.scatter(*xy[mask].T, s=22, facecolors="none",
                       edgecolors=np.where(value >= 0, "#BD5E66", "#417F95"),
                       linewidths=width, rasterized=True, zorder=3)
        maximum = max(float(shown[quantity].max()), 1e-12) if len(shown) else 1.
        for row in shown.itertuples():
            ax.add_patch(FancyArrowPatch(
                xy[row.sender], xy[row.receiver], arrowstyle="-|>", connectionstyle="arc3,rad=.12",
                shrinkA=2.2, shrinkB=2.6, mutation_scale=5,
                lw=.2 + 1.2 * getattr(row, quantity) / maximum,
                color=colors[row.module_name], alpha=.8, zorder=4,
            ))
        if len(shown):
            sources = shown.sender.unique().astype(int)
            ax.scatter(*xy[sources].T, s=27, facecolors="none", edgecolors="#486970",
                       linewidths=.7, rasterized=True, zorder=5)
        if focus_receiver is not None:
            ax.annotate("R", xy[focus_receiver], xytext=(4, 4), textcoords="offset points", fontsize=6, zorder=6)
        viewport = xy[np.unique(np.r_[np.flatnonzero(mask), shown.sender.to_numpy(int)])]
        center = (viewport.max(0) + viewport.min(0)) / 2
        span = max(float(np.ptp(viewport, axis=0).max()), 1.) * 1.13
        ax.set(xlim=(center[0] - span / 2, center[0] + span / 2),
               ylim=(center[1] - span / 2, center[1] + span / 2), aspect="equal",
               title="Spatial sender catchment")
        if invert_y:
            ax.invert_yaxis()
        ax.axis("off")
        handles = [Line2D([0], [0], color=colors[name], label=name.replace("->", "–"))
                   for name in shown.module_name.drop_duplicates()]
        if "velocity_value" in sites:
            handles.extend([
                Line2D([0], [0], color=color, marker="o", markerfacecolor="none", linestyle="none",
                       label=f"{result['metadata']['velocity_label']} {sign}")
                for color, sign in (("#417F95", "−"), ("#BD5E66", "+"))
            ])
        if handles:
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.04),
                      ncol=2, frameon=False, fontsize=5)
        # Keep the exact drawing subset discoverable without mutating result.
        ax.cellvocal_drawn_event_indices = shown.event_index.to_numpy(int)
        style_communication_figure(fig)
    return fig, ax
