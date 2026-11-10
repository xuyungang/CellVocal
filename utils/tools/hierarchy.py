"""Exact receiver attribution and spatial communication summaries.

These tools consume fitted CellVocal outputs, never refit a model and never
interpret an inferred signalling influence as a causal effect. A sender's signed contribution
is its fraction of the corresponding receiver/module RNA-velocity response.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from utils.tools.data import (
    aligned_vector, communication_events, incoming_influence, receiver_mask_array,
)


def _dense(values: Any) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def _receiver_index(adata: Any, receiver: int | str) -> int:
    if isinstance(receiver, str):
        if not adata.obs_names.is_unique:
            raise ValueError("Observation names must be unique for name-based selection.")
        if receiver not in adata.obs_names:
            raise KeyError(f"Unknown receiver: {receiver}")
        return int(adata.obs_names.get_loc(receiver))
    if not isinstance(receiver, (int, np.integer)) or not 0 <= receiver < adata.n_obs:
        raise ValueError("receiver must be a valid observation index or name.")
    return int(receiver)


def _components(
    adata: Any, names: Sequence[str], basis_s: Any, mixing_lambda: Any,
    ccc_velocity_scale: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    if isinstance(basis_s, pd.DataFrame):
        if not basis_s.index.is_unique or not basis_s.columns.is_unique:
            raise ValueError("The response basis must have unique module and gene labels.")
        basis = basis_s.loc[list(names), list(adata.var_names)].to_numpy(float)
    else:
        basis = np.asarray(basis_s, dtype=float)
    if basis.shape != (len(names), adata.n_vars) or not np.isfinite(basis).all():
        raise ValueError("basis_s must be finite and aligned to all modules and adata.var_names.")
    lam = (
        aligned_vector(mixing_lambda, adata.obs_names, "mixing_lambda")
        if isinstance(mixing_lambda, pd.Series) else np.asarray(mixing_lambda, dtype=float)
    )
    if lam.ndim == 0:
        lam = np.full(adata.n_obs, float(lam))
    lam = lam.reshape(-1)
    if lam.shape != (adata.n_obs,) or not np.isfinite(lam).all() or (lam < 0).any():
        raise ValueError("mixing_lambda must contain one finite nonnegative value per observation.")
    gamma = float(ccc_velocity_scale)
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("ccc_velocity_scale must be finite and nonnegative.")
    return basis, lam, gamma


def receiver_response(
    adata: Any,
    receiver: int | str,
    *,
    basis_s: Any,
    mixing_lambda: Any,
    ccc_velocity_scale: float,
    genes: Sequence[str] | None = None,
    gene_scale: Sequence[float] | pd.Series | None = None,
    prefix: str = "cv",
    check_reconstruction: bool = True,
    rtol: float = 2e-5,
    atol: float = 2e-7,
) -> dict[str, Any]:
    """Resolve one receiver's communication response into modules and senders.

    ``basis_s`` is the spliced response basis, shaped (modules, adata.n_vars),
    in exactly the fitted gene order. For the standard checkpoint it is
    ``model.ccc_param_basis[:, :, 1]``. The returned contribution is
    ``ccc_velocity_scale * mixing_lambda[r] * q[r,m] * basis_s[m,g]``;
    multiplying by the sender fraction yields each event's contribution.

    All modules are retained, including those omitted from a later plot.
    ``gene_scale`` optionally divides each selected gene by a supplied positive
    input-expression standard deviation. Reconstruction is checked in the
    original units against ``layers[prefix + '_ccc_velocity']`` before scaling.
    Total and intrinsic velocities are reported separately when available.
    This is an attribution of the fitted model, not an intervention.
    """
    r = _receiver_index(adata, receiver)
    incoming = incoming_influence(adata, prefix=prefix)
    names = incoming.columns.astype(str).tolist()
    basis, lam, gamma = _components(adata, names, basis_s, mixing_lambda, ccc_velocity_scale)
    if not adata.var_names.is_unique:
        raise ValueError("Gene names must be unique.")
    selected = list(adata.var_names if genes is None else genes)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("genes must be a nonempty sequence of unique names.")
    indices = adata.var_names.get_indexer(selected)
    if (indices < 0).any():
        raise KeyError(f"Genes absent from fitted data: {np.asarray(selected)[indices < 0].tolist()}")
    if gene_scale is None:
        scales = np.ones(len(selected))
    elif isinstance(gene_scale, pd.Series):
        scales = gene_scale.loc[selected].to_numpy(float)
    else:
        scales = np.asarray(gene_scale, dtype=float)
    if scales.shape != (len(selected),) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("gene_scale must contain a positive finite scale per selected gene.")
    q = incoming.iloc[r].to_numpy(float)
    raw = gamma * lam[r] * q[:, None] * basis[:, indices]
    reconstructed = raw.sum(axis=0)
    layer = f"{prefix}_ccc_velocity"
    measured = None
    if layer in adata.layers:
        measured = _dense(adata.layers[layer][r:r + 1, indices]).reshape(-1).astype(float)
        if not np.isfinite(measured).all():
            raise ValueError("The saved communication velocity contains nonfinite values.")
        if check_reconstruction and not np.allclose(reconstructed, measured, rtol=rtol, atol=atol):
            raise ValueError(
                "Response basis does not reconstruct the saved communication velocity; "
                f"maximum absolute error {np.max(np.abs(reconstructed - measured)):g}."
            )
    elif check_reconstruction:
        raise KeyError(f"Missing {layer!r}; supply the saved layer or set check_reconstruction=False.")
    values = raw / scales[None, :]
    module_table = pd.DataFrame({
        "receiver": r, "receiver_id": str(adata.obs_names[r]),
        "module": np.repeat(np.arange(len(names)), len(selected)),
        "module_name": np.repeat(names, len(selected)),
        "gene": np.tile(selected, len(names)),
        "module_activation_prob": np.repeat(q, len(selected)),
        "mixing_lambda": float(lam[r]), "ccc_velocity_scale": gamma,
        "gene_scale": np.tile(scales, len(names)),
        "contribution": values.ravel(),
    })
    events = communication_events(adata, prefix=prefix)
    events = events.loc[events.receiver.eq(r)].copy()
    events["sender_id"] = np.asarray(adata.obs_names, str)[events.sender.to_numpy(int)]
    events["receiver_id"] = str(adata.obs_names[r])
    event_values = (
        values[events.module.to_numpy(int)] *
        events.sender_share.to_numpy(float)[:, None]
    )
    event_response = events.loc[events.index.repeat(len(selected))].reset_index(drop=True)
    event_response["gene"] = np.tile(selected, len(events))
    event_response["gene_scale"] = np.tile(scales, len(events))
    event_response["contribution"] = event_values.ravel()
    event_sum = np.zeros_like(values)
    np.add.at(event_sum, events.module.to_numpy(int), event_values)
    if not np.allclose(event_sum, values, rtol=rtol, atol=atol):
        raise ValueError("Sender contributions do not sum to the receiver/module responses.")
    summary = pd.DataFrame({
        "receiver": r, "receiver_id": str(adata.obs_names[r]),
        "gene": selected, "gene_scale": scales,
        "ccc_velocity_reconstructed": reconstructed / scales,
    })
    if measured is not None:
        summary["ccc_velocity_saved"] = measured / scales
    for output, key in (
        ("total_velocity", f"{prefix}_velocity"),
        ("intrinsic_velocity", f"{prefix}_intrinsic_velocity"),
    ):
        if key in adata.layers:
            summary[output] = _dense(adata.layers[key][r:r + 1, indices]).reshape(-1) / scales
    if check_reconstruction and {"total_velocity", "intrinsic_velocity"}.issubset(summary.columns):
        total = summary.total_velocity.to_numpy() * scales
        intrinsic = summary.intrinsic_velocity.to_numpy() * scales
        if not np.isfinite(total).all() or not np.isfinite(intrinsic).all():
            raise ValueError("Total or intrinsic velocity contains nonfinite values.")
        if not np.allclose(total, intrinsic + reconstructed, rtol=rtol, atol=atol):
            raise ValueError("Saved total velocity does not reconstruct as intrinsic plus communication velocity.")
    return {
        "events": events.reset_index(drop=True),
        "event_response": event_response,
        "module_response": module_table,
        "gene_summary": summary,
        "metadata": {
            "receiver": r, "receiver_id": str(adata.obs_names[r]),
            "prefix": prefix, "units": "input SD" if gene_scale is not None else "model RNA units",
            "reconstruction_checked": check_reconstruction,
            "max_reconstruction_error": (
                float(np.max(np.abs(reconstructed - measured))) if measured is not None else None
            ),
        },
    }


def tissue_communication(
    adata: Any,
    *,
    modules: Sequence[str | int] | None = None,
    receiver_mask: Sequence[bool] | None = None,
    source_labels: str | Sequence[str] | None = None,
    observed_values: Sequence[float] | None = None,
    velocity_values: Sequence[float] | None = None,
    velocity_label: str = "Total RNA velocity",
    observed_label: str = "Current expression",
    basis_s: Any | None = None,
    mixing_lambda: Any | None = None,
    ccc_velocity_scale: float | None = None,
    gene_weights: Sequence[float] | None = None,
    spatial_key: str = "spatial",
    prefix: str = "cv",
) -> dict[str, Any]:
    """Summarize all selected routes without constructing a dense spatial graph.

    ``receiver_mask`` selects evaluation sites while source sites remain eligible
    throughout the tissue. Positive inferred signalling influences determine source shares.
    If all response components are supplied, routes also contain a signed
    programme contribution (when ``gene_weights`` is supplied) and a nonnegative
    RMS response magnitude. No outcome-driven filtering or route sampling occurs
    here; plotting can limit displayed routes without changing these tables.
    Observation labels and values must follow the full AnnData observation order.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"Missing obsm[{spatial_key!r}].")
    xy = np.asarray(adata.obsm[spatial_key], dtype=float)
    if xy.ndim != 2 or xy.shape[0] != adata.n_obs or xy.shape[1] < 2 or not np.isfinite(xy[:, :2]).all():
        raise ValueError("Spatial coordinates must contain two finite coordinates per observation.")
    xy = xy[:, :2]
    mask = receiver_mask_array(receiver_mask, adata.obs_names)
    all_events = communication_events(adata, prefix=prefix)
    names = np.asarray(adata.uns[f"{prefix}_ccc"]["lr_pairs"], dtype=str)
    if modules is None:
        selected = np.arange(len(names))
    else:
        selected_list = []
        for module in modules:
            if isinstance(module, str):
                found = np.flatnonzero(names == module)
                if len(found) != 1:
                    raise KeyError(f"Unknown or ambiguous module: {module}")
                selected_list.append(int(found[0]))
            elif isinstance(module, (int, np.integer)) and 0 <= module < len(names):
                selected_list.append(int(module))
            else:
                raise ValueError(f"Invalid module: {module}")
        selected = np.unique(selected_list)
    routes = all_events.loc[
        all_events.module.isin(selected) &
        mask[all_events.receiver.to_numpy(int)] &
        all_events.event_influence.gt(0)
    ].copy().reset_index(drop=True)
    labels = np.full(adata.n_obs, "All sources", dtype=object) if source_labels is None else aligned_vector(
        adata.obs[source_labels] if isinstance(source_labels, str) else source_labels,
        adata.obs_names, "source_labels", dtype=None,
    )
    if labels.shape != (adata.n_obs,) or pd.isna(labels).any():
        raise ValueError("source_labels must contain a nonmissing label per observation.")
    labels = labels.astype(str)
    si, ri = routes.sender.to_numpy(int), routes.receiver.to_numpy(int)
    routes["source_label"] = labels[si]
    routes["sender_id"] = np.asarray(adata.obs_names, str)[si]
    routes["receiver_id"] = np.asarray(adata.obs_names, str)[ri]
    for key, values in (("sender_x", xy[si, 0]), ("sender_y", xy[si, 1]),
                        ("receiver_x", xy[ri, 0]), ("receiver_y", xy[ri, 1])):
        routes[key] = values
    sites = pd.DataFrame({
        "site": np.arange(adata.n_obs), "site_id": np.asarray(adata.obs_names, str),
        "x": xy[:, 0], "y": xy[:, 1], "is_receiver": mask, "source_label": labels,
        "incoming_influence": np.bincount(ri, weights=routes.event_influence, minlength=adata.n_obs),
    })
    for name, values in (("observed_value", observed_values), ("velocity_value", velocity_values)):
        if values is not None:
            values = aligned_vector(values, adata.obs_names, name)
            if values.shape != (adata.n_obs,) or not np.isfinite(values[mask]).all():
                raise ValueError(f"{name} must be finite at all selected receivers and have length n_obs.")
            sites[name] = values
    components = (basis_s, mixing_lambda, ccc_velocity_scale)
    if any(item is not None for item in components):
        if not all(item is not None for item in components):
            raise ValueError("Provide basis_s, mixing_lambda and ccc_velocity_scale together.")
        basis, lam, gamma = _components(adata, names, *components)
        amplitude = gamma * lam[ri] * routes.module_activation_prob.to_numpy(float) * routes.sender_share.to_numpy(float)
        routes["response_rms"] = amplitude * np.sqrt(np.mean(basis ** 2, axis=1))[routes.module.to_numpy(int)]
        if gene_weights is not None:
            weights = aligned_vector(gene_weights, adata.var_names, "gene_weights")
            if weights.shape != (adata.n_vars,) or not np.isfinite(weights).all():
                raise ValueError("gene_weights must be finite and aligned to adata.var_names.")
            routes["programme_contribution"] = amplitude * (basis @ weights)[routes.module.to_numpy(int)]
    elif gene_weights is not None:
        raise ValueError("gene_weights requires all three fitted response components.")
    aggregations = {
        "total_influence": ("event_influence", "sum"),
        "n_events": ("event_index", "size"), "n_senders": ("sender", "nunique"),
        "n_receivers": ("receiver", "nunique"),
    }
    for name in ("response_rms", "programme_contribution"):
        if name in routes:
            aggregations[name] = (name, "sum")
    source_summary = routes.groupby(["module_name", "source_label"], sort=False).agg(**aggregations).reset_index()
    totals = source_summary.groupby("module_name").total_influence.transform("sum")
    source_summary["influence_fraction"] = source_summary.total_influence / totals
    return {
        "sites": sites, "routes": routes, "source_summary": source_summary,
        "metadata": {
            "prefix": prefix, "modules": names[selected].tolist(),
            "observed_label": observed_label, "velocity_label": velocity_label,
            "route_quantity": "response_rms" if "response_rms" in routes else "event_influence",
            "n_receivers": int(mask.sum()),
            "n_input_positive_receivers": int(routes.receiver.nunique()),
        },
    }
