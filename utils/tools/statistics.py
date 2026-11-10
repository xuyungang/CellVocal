"""Conserved source -> section -> receiver-state summaries of communication input."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import aligned_vector, communication_events, receiver_mask_array


def _observation_values(specification, sample, adata, name, *, dtype=None):
    if isinstance(specification, str):
        values = adata.obs[specification]
    elif isinstance(specification, Mapping):
        values = specification[sample]
    else:
        raise TypeError(f"{name} must be an obs column name or a section-to-vector mapping")
    return aligned_vector(values, adata.obs_names, name, dtype=dtype)


def communication_statistics(
    sections: Mapping,
    module: str | Sequence[str],
    *,
    source_labels,
    state_values,
    receiver_masks: Mapping | None = None,
    donors: Mapping | None = None,
    weighting: str = "section",
    source_groups: Mapping[str, str] | None = None,
    source_order: Sequence[str] | None = None,
    other_source: str = "Other",
    state_labels: Sequence[str] = ("Low", "Intermediate", "High"),
    quantiles: Sequence[float] = (1 / 3, 2 / 3),
    state_thresholds: Sequence[float] | None = None,
    prefix: str = "cv",
) -> dict:
    """Summarize inferred signalling influence by source, section and state.

    ``sections`` maps section identifiers to full-observation fitted AnnData
    objects. Only obs and uns[prefix + '_ccc'] are needed. ``source_labels`` and
    ``state_values`` are obs column names, or mappings from section to aligned
    observation vectors. The state readout is measured current state, not a
    velocity sign, effect size or significance score.

    Every selected event contributes ``q * sender_share * section_weight /
    n_receivers``. All eligible receivers, including zero-input sites, enter the
    denominator. Section weighting gives each section 1 / n_sections. Donor
    weighting gives each donor equal weight, shared across its technical
    sections. Supply actual donor IDs explicitly for ``weighting='donor'``.

    By default, shared state thresholds are quantiles of eligible receiver
    values using the same section/receiver weights. Ties are assigned to the
    higher interval and may leave an interval empty; thresholds are never
    shifted to improve a result. Explicit nondecreasing thresholds are supported.

    Returns the full source/section/state flow table, its tensor, section
    summaries, receiver states and threshold provenance. Both sides of a
    Sankey plot are marginals of this same tensor; no input is duplicated.
    """
    if not isinstance(sections, Mapping) or not sections:
        raise ValueError("sections must be a nonempty mapping of section ID to AnnData")
    samples = list(sections)
    if any(not isinstance(s, str) or not s for s in samples):
        raise ValueError("Section IDs must be nonempty strings")
    modules = [module] if isinstance(module, str) else list(module)
    if not modules or len(set(modules)) != len(modules):
        raise ValueError("Specify one or more unique module names")
    if weighting not in {"section", "donor"}:
        raise ValueError("weighting must be 'section' or 'donor'")
    if weighting == "donor" and donors is None:
        raise ValueError("Donor weighting requires explicit biological donor IDs")
    for name, mapping in (("donors", donors), ("receiver_masks", receiver_masks)):
        if mapping is not None and (not isinstance(mapping, Mapping) or set(mapping) != set(samples)):
            raise ValueError(f"{name} must contain exactly the selected section IDs")
    for name, specification in (("source_labels", source_labels), ("state_values", state_values)):
        if isinstance(specification, Mapping) and set(specification) != set(samples):
            raise ValueError(f"{name} must contain exactly the selected section IDs")
    if donors is not None and any(pd.isna(donors[s]) for s in samples):
        raise ValueError("Donor IDs cannot be missing")
    labels = list(state_labels)
    probabilities = np.asarray(quantiles, float)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("State labels must be nonempty and unique")
    if (probabilities.ndim != 1 or len(probabilities) != len(labels) - 1
            or not np.isfinite(probabilities).all() or (probabilities <= 0).any()
            or (probabilities >= 1).any() or (np.diff(probabilities) <= 0).any()):
        raise ValueError("quantiles must be increasing probabilities, one fewer than state_labels")
    if weighting == "donor":
        counts = pd.Series([donors[s] for s in samples]).value_counts()
        section_weights = {s: 1 / (len(counts) * counts.loc[donors[s]]) for s in samples}
    else:
        section_weights = {s: 1 / len(samples) for s in samples}

    prepared, values, weights, receiver_frames = {}, [], [], []
    observed_sources = []
    for sample, adata in sections.items():
        if not adata.obs_names.is_unique:
            raise ValueError(f"Section {sample!r} has duplicate observation IDs")
        mask = receiver_mask_array(None if receiver_masks is None else receiver_masks[sample], adata.obs_names)
        n = int(mask.sum())
        if n == 0:
            raise ValueError(f"Section {sample!r} has no selected receivers")
        sources = _observation_values(source_labels, sample, adata, "source_labels")
        state = _observation_values(state_values, sample, adata, "state_values", dtype=float)
        if not np.isfinite(state[mask]).all():
            raise ValueError(f"Section {sample!r} has nonfinite state values at selected receivers")
        events = communication_events(adata, prefix=prefix)
        available = set(np.asarray(adata.uns[f"{prefix}_ccc"]["lr_pairs"], str))
        absent = set(modules) - available
        if absent:
            raise KeyError(f"Modules absent from section {sample!r}: {sorted(absent)}")
        events = events.loc[events.module_name.isin(modules) &
                            mask[events.receiver.to_numpy(int)]].copy()
        sender_sources = sources[events.sender.to_numpy(int)]
        if pd.isna(sender_sources).any():
            raise ValueError(f"Section {sample!r} has missing source labels for selected events")
        sender_sources = sender_sources.astype(str)
        if source_groups is not None:
            sender_sources = np.asarray([source_groups.get(s, other_source) for s in sender_sources], str)
        events["source"] = sender_sources
        observed_sources.extend(sender_sources.tolist())
        # Use the exported factors in float64, matching the exact input ledger.
        events["raw_event_influence"] = events.module_activation_prob * events.sender_share
        prepared[sample] = (adata, mask, state, events)
        values.append(state[mask])
        weights.append(np.full(n, section_weights[sample] / n))

    if source_order is None:
        sources = list(dict.fromkeys(observed_sources))
    else:
        sources = list(source_order)
        if len(set(sources)) != len(sources) or set(observed_sources) - set(sources):
            raise ValueError("source_order must be unique and include every retained source group")
    if not sources:
        sources = [other_source]
    if state_thresholds is None:
        ordered = np.argsort(np.concatenate(values))
        sorted_values = np.concatenate(values)[ordered]
        sorted_weights = np.concatenate(weights)[ordered]
        positions = (np.cumsum(sorted_weights) - sorted_weights / 2) / sorted_weights.sum()
        cuts = np.interp(probabilities, positions, sorted_values)
        threshold_method = "weighted quantiles across all eligible receivers"
    else:
        cuts = np.asarray(state_thresholds, float)
        if (cuts.shape != probabilities.shape or not np.isfinite(cuts).all()
                or (np.diff(cuts) < 0).any()):
            raise ValueError("state_thresholds must be finite nondecreasing values matching quantiles")
        threshold_method = "caller-supplied thresholds"
    rows, section_rows = [], []
    for sample, (adata, mask, state, events) in prepared.items():
        n = int(mask.sum())
        per_receiver_weight = section_weights[sample] / n
        # Values outside the receiver population are not used for thresholds or flow.
        event_states = np.searchsorted(cuts, state[events.receiver.to_numpy(int)], side="right")
        events["state"] = np.asarray(labels, str)[event_states]
        events["weighted_influence"] = events.raw_event_influence * per_receiver_weight
        grouped = events.groupby(["source", "module_name", "state"], observed=True).agg(
            raw_influence_sum=("raw_event_influence", "sum"),
            weighted_influence=("weighted_influence", "sum"),
            n_events=("raw_event_influence", lambda x: int((x > 0).sum())),
        )
        complete = pd.MultiIndex.from_product([sources, modules, labels],
                                              names=["source", "module_name", "state"])
        grouped = grouped.reindex(complete, fill_value=0).reset_index()
        grouped["section"], grouped["donor"] = sample, None if donors is None else donors[sample]
        rows.append(grouped)
        incoming = np.bincount(events.receiver.to_numpy(int),
                              weights=events.raw_event_influence, minlength=adata.n_obs)
        independent_q = events.drop_duplicates(["receiver", "module"]).module_activation_prob.sum()
        expected = float(independent_q * per_receiver_weight)
        actual = float(grouped.weighted_influence.sum())
        if not np.isclose(actual, expected, rtol=2e-5, atol=1e-8):
            raise ValueError(f"Section {sample!r} fails influence conservation")
        section_rows.append(dict(
            section=sample, donor=None if donors is None else donors[sample],
            n_receivers=n, n_active_receivers=int((incoming[mask] > 0).sum()),
            active_fraction=float((incoming[mask] > 0).mean()),
            section_weight=section_weights[sample], receiver_weight=per_receiver_weight,
            mean_incoming_influence=float(incoming[mask].mean()), section_total_influence=actual,
            conservation_error=actual - expected,
        ))
        receiver_frames.append(pd.DataFrame({
            "section": sample, "observation": adata.obs_names[mask],
            "observed_state": state[mask],
            "state": np.asarray(labels, str)[np.searchsorted(cuts, state[mask], side="right")],
            "incoming_influence": incoming[mask], "receiver_weight": per_receiver_weight,
        }))
    flow = pd.concat(rows, ignore_index=True)
    full_index = pd.MultiIndex.from_product([sources, samples, labels],
                                           names=["source", "section", "state"])
    tensor = flow.groupby(["source", "section", "state"], observed=True).weighted_influence.sum()
    tensor = tensor.reindex(full_index, fill_value=0).to_numpy().reshape(len(sources), len(samples), len(labels))
    return dict(
        flow=flow, tensor=tensor, sections=pd.DataFrame(section_rows),
        receivers=pd.concat(receiver_frames, ignore_index=True),
        thresholds=pd.DataFrame({"quantile": probabilities, "value": cuts, "method": threshold_method}),
        metadata=dict(modules=modules, source_order=sources, section_order=samples,
                      state_order=labels, weighting=weighting,
                      width_label=f"{'Donor' if weighting == 'donor' else 'Section'}-balanced influence",
                      interpretation="Inferred nonnegative signalling influence; not signed RNA velocity or a causal effect"),
    )
