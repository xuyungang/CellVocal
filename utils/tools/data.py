"""Validated adapters for fitted CellVOCAL communication and RNA dynamics.

These tools read fitted outputs without changing them or importing paper analyses.
Inferred signalling influence is q(receiver, module) * p(sender | receiver, module).
Event influence is nonnegative. Multiplication by a signed gene-effect coefficient
produces a gene contribution to RNA velocity.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def aligned_vector(values, index: pd.Index, name: str, *, dtype=float) -> np.ndarray:
    """Validate positional vectors or align a Series with exactly matching IDs."""
    if isinstance(values, pd.Series):
        if not values.index.is_unique or not index.is_unique:
            raise ValueError(f"{name} requires unique observation IDs")
        if len(values) != len(index) or not index.isin(values.index).all():
            raise ValueError(f"{name} Series must contain exactly the observation IDs")
        values = values.reindex(index).to_numpy()
    result = np.asarray(values)
    if result.ndim != 1 or len(result) != len(index):
        raise ValueError(f"{name} must have one value per observation")
    return result if dtype is None else result.astype(dtype)


def receiver_mask_array(mask, index: pd.Index) -> np.ndarray:
    if mask is None:
        return np.ones(len(index), dtype=bool)
    values = aligned_vector(mask, index, "receiver_mask", dtype=None)
    if values.dtype.kind != "b":
        raise ValueError("receiver_mask must be Boolean, not observation indices")
    return values


def communication_events(adata, *, prefix: str = "cv") -> pd.DataFrame:
    """Return validated events in original order, including zero-influence events.

    Integer indices refer to the *original full* AnnData. Subsetting AnnData does
    not remap indices stored in ``uns``; use a receiver mask in downstream tools.
    If only event_influence is present, q is computed within receiver/module
    groups. Exports containing q and p are checked against a = q * p.
    """
    key = f"{prefix}_ccc"
    if key not in adata.uns:
        raise KeyError(f"Missing uns[{key!r}]")
    ccc = adata.uns[key]
    names = np.asarray(ccc["lr_pairs"], dtype=str)
    if names.ndim != 1 or pd.Index(names).has_duplicates:
        raise ValueError("lr_pairs must be a one-dimensional set of unique module names")
    arrays = {}
    for name, bound in (("sender", adata.n_obs), ("receiver", adata.n_obs),
                        ("module", len(names))):
        value = np.asarray(ccc[name])
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite integer vector")
        if not np.equal(value, np.floor(value)).all() or ((value < 0) | (value >= bound)).any():
            raise ValueError(f"{name} contains invalid indices for this full AnnData")
        arrays[name] = value.astype(np.int64)
    n = len(arrays["module"])
    if any(len(v) != n for v in arrays.values()):
        raise ValueError("CCC event arrays must have matching lengths")
    group = arrays["receiver"] * len(names) + arrays["module"]
    _, first, inverse = np.unique(group, return_index=True, return_inverse=True)

    def nonnegative(name):
        value = np.asarray(ccc[name], dtype=float)
        if value.shape != (n,) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must be a finite nonnegative vector of length {n}")
        return value

    if "event_influence" in ccc:
        event_influence = nonnegative("event_influence")
    elif "module_activation_prob" in ccc and "sender_share" in ccc:
        event_influence = nonnegative("module_activation_prob") * nonnegative("sender_share")
    else:
        raise KeyError("CCC requires event_influence or both module_activation_prob and sender_share")
    total = np.bincount(inverse, weights=event_influence, minlength=len(first))
    q = total[inverse]
    if "module_activation_prob" in ccc:
        module_activation_prob = nonnegative("module_activation_prob")
        if not np.allclose(module_activation_prob, module_activation_prob[first][inverse], rtol=1e-5, atol=1e-7):
            raise ValueError("module_activation_prob differs within a receiver/module group")
        if not np.allclose(q, module_activation_prob, rtol=1e-5, atol=1e-7):
            raise ValueError("Summed event influence does not reconstruct module_activation_prob")
        q = module_activation_prob
    fraction = np.divide(event_influence, q, out=np.zeros(n), where=q > 0)
    if "sender_share" in ccc:
        fraction = nonnegative("sender_share")
        if not np.allclose(np.bincount(inverse, weights=fraction), 1., rtol=1e-5, atol=1e-6):
            raise ValueError("Conditional sender shares must sum to one within receiver/module groups")
        if not np.allclose(event_influence, q * fraction, rtol=1e-5, atol=1e-7):
            raise ValueError("event_influence does not equal module_activation_prob * sender_share")
    return pd.DataFrame({
        "event_index": np.arange(n), **arrays, "module_name": names[arrays["module"]],
        "event_influence": event_influence, "sender_share": fraction, "module_activation_prob": q,
    })


def incoming_influence(adata, *, prefix: str = "cv") -> pd.DataFrame:
    """Sum event influences over all candidate senders for each receiver/module.

    This equals module_activation_prob when all senders are retained. Storage
    is O(receivers * modules), never O(receivers²)."""
    events = communication_events(adata, prefix=prefix)
    names = np.asarray(adata.uns[f"{prefix}_ccc"]["lr_pairs"], str)
    values = np.zeros((adata.n_obs, len(names)))
    # All senders are retained, so their event influences sum to the shared q.
    unique = events.drop_duplicates(["receiver", "module"])
    values[unique.receiver.to_numpy(int), unique.module.to_numpy(int)] = unique.module_activation_prob
    return pd.DataFrame(values, index=adata.obs_names, columns=names)


def _dense_columns(matrix, indices):
    result = matrix[:, indices]
    return result.toarray().astype(float) if sparse.issparse(result) else np.asarray(result, float)


def program_scores(
    adata, genes: Sequence[str], *, prefix: str = "cv", expression_layer: str | None = "spliced",
    gene_scale: pd.Series | None = None, missing: str = "raise",
    denominator: int | None = None,
) -> pd.DataFrame:
    """Project expression and velocity onto the same fixed gene average.

    ``gene_scale`` optionally provides positive expression SDs keyed by gene.
    No velocity-specific z-scoring, centering or sign reversal is performed.
    With ``missing='omit'`` the original signature denominator is retained and
    gene coverage is recorded in ``attrs``; this is a partial projection, not
    an assertion that absent genes have zero biological response.
    """
    genes = list(genes)
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("genes must be nonempty and unique")
    if not adata.var_names.is_unique:
        raise ValueError("AnnData gene names must be unique")
    if missing not in {"raise", "omit"}:
        raise ValueError("missing must be 'raise' or 'omit'")
    absent = [g for g in genes if g not in adata.var_names]
    if absent and missing == "raise":
        raise KeyError(f"Signature genes unavailable: {absent}")
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise ValueError("No signature genes are available")
    denom = len(genes) if denominator is None else denominator
    if not isinstance(denom, (int, np.integer)) or denom < len(present):
        raise ValueError("denominator must be an integer at least the number of available genes")
    ix = adata.var_names.get_indexer(present)
    scale = np.ones(len(present))
    if gene_scale is not None:
        if not isinstance(gene_scale, pd.Series) or not gene_scale.index.is_unique:
            raise ValueError("gene_scale must be a uniquely gene-indexed Series")
        scale = gene_scale.reindex(present).to_numpy(float)
        if not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Every available signature gene needs a finite positive scale")
    expression = adata.X if expression_layer is None else adata.layers[expression_layer]
    matrices = {"current_expression": expression, "total_velocity": adata.layers[f"{prefix}_velocity"]}
    for short in ("ccc", "intrinsic"):
        if f"{prefix}_{short}_velocity" in adata.layers:
            matrices[f"{short}_velocity"] = adata.layers[f"{prefix}_{short}_velocity"]
    values = {}
    for label, matrix in matrices.items():
        array = _dense_columns(matrix, ix)
        if not np.isfinite(array).all():
            raise ValueError(f"{label} contains nonfinite signature values")
        values[label] = (array / scale).sum(axis=1) / denom
    result = pd.DataFrame(values, index=adata.obs_names)
    result.attrs.update(genes_requested=genes, genes_available=present, genes_missing=absent,
                        denominator=int(denom), expression_layer=expression_layer,
                        units="input SD" if gene_scale is not None else "model input",
                        velocity_definition="signed total spliced RNA velocity")
    return result


def load_response_components(adata, checkpoint_path: str | Path, *, prefix="cv", batch_size=1024):
    """Recover fixed decoder components from a matching fitted checkpoint on CPU.

    This is inference only, with no retraining or disk cache. Full gene/module
    axes must match the fitted output; the receiver tool additionally checks
    the reconstructed CCC velocity against the stored velocity layer.
    """
    import torch
    from model.cellvocal import RNAVelocityHead
    from model.train.config import CellVOCALModelConfig
    from model.train.checkpoint import CHECKPOINT_VERSION

    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version: {payload.get('version')!r}")
    names = np.asarray(adata.uns[f"{prefix}_ccc"]["lr_pairs"], str)
    if (payload["n_genes"], payload["n_modules"]) != (adata.n_vars, len(names)):
        raise ValueError("Checkpoint gene/module dimensions do not match the full AnnData")
    # The fitted latent states already exist. Load only the decoder, so optional
    # lazily-created encoder annotation embeddings do not need reconstruction.
    cfg = CellVOCALModelConfig(**payload["config"]["model"])
    head = RNAVelocityHead(adata.n_vars, cfg)
    state = {k.removeprefix("kinetic_head."): v for k, v in payload["model_state_dict"].items()
             if k.startswith("kinetic_head.")}
    head.load_state_dict(state, strict=True)
    head.eval()
    latent = np.asarray(adata.obsm[f"{prefix}_latent"], np.float32)
    time = adata.obs[f"{prefix}_time"].to_numpy(np.float32)
    if not np.isfinite(latent).all() or not np.isfinite(time).all():
        raise ValueError("Latent states and time must be finite")
    lam = []
    with torch.no_grad():
        for start in range(0, adata.n_obs, batch_size):
            z = torch.from_numpy(latent[start:start + batch_size])
            t = torch.from_numpy(time[start:start + batch_size])
            u, s, _, _ = head.intrinsic_dynamics(z, t)
            lam.append(head.mixing_lambda(u, s, z).cpu().numpy().reshape(-1))
        basis = payload["model_state_dict"]["ccc_param_basis"][:, :, 1].detach().cpu().numpy()
    return {"basis_s": basis, "mixing_lambda": np.concatenate(lam),
            "ccc_velocity_scale": float(cfg.ccc_velocity_scale)}
