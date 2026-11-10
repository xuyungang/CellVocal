"""Quality control for AnnData objects used by RNA-velocity models."""
from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import anndata as ad
import numpy as np
from scipy import sparse

_QC_PREFIX = "cellvocal_qc"


def _as_layer(adata: ad.AnnData, layer: str):
    if layer not in adata.layers:
        raise KeyError(f"AnnData must have layers[{layer!r}] for RNA velocity QC")
    x = adata.layers[layer]
    if sparse.issparse(x):
        return x.tocsr()
    return np.asarray(x)


def _sum_axis(x, axis: int) -> np.ndarray:
    return np.asarray(x.sum(axis=axis)).ravel().astype(np.float64, copy=False)


def _nnz_axis(x, axis: int) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.getnnz(axis=axis)).ravel().astype(np.int64, copy=False)
    return np.count_nonzero(np.asarray(x) > 0, axis=axis).astype(np.int64, copy=False)


def _finite_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=np.float64)).all(axis=1)


def _prefixed(name: str) -> str:
    return f"{_QC_PREFIX}_{name}"


def _normalize_layers_by_spot(adata: ad.AnnData, *, layers: Sequence[str]) -> float:
    """Scale each spot so total spliced+unspliced counts match the median spot depth."""
    total = np.zeros(adata.n_obs, dtype=np.float64)
    layer_mats = []
    for layer in layers:
        x = _as_layer(adata, layer)
        layer_mats.append((layer, x))
        total += _sum_axis(x, axis=1)

    positive = total > 0
    if not positive.any():
        raise ValueError("Cannot normalize RNA velocity layers: all spots have zero counts")
    resolved_target = float(np.median(total[positive]))
    scale = np.ones_like(total, dtype=np.float64)
    scale[positive] = resolved_target / total[positive]
    scale[~positive] = 0.0

    for layer, x in layer_mats:
        if sparse.issparse(x):
            adata.layers[layer] = sparse.diags(scale, format="csr") @ x
        else:
            adata.layers[layer] = np.asarray(x, dtype=np.float64) * scale[:, None]
    return resolved_target


def _top_gene_mask_by_mean(counts: np.ndarray, n_top_genes: int) -> np.ndarray:
    if n_top_genes >= counts.size:
        return np.ones(counts.size, dtype=bool)
    keep = np.zeros(counts.size, dtype=bool)
    top_idx = np.argpartition(-counts, n_top_genes - 1)[:n_top_genes]
    keep[top_idx] = True
    return keep


def _remap_ground_truth_after_obs_filter(
    adata: ad.AnnData,
    *,
    kept_obs_indices: np.ndarray,
    n_obs_before: int,
) -> None:
    """Keep ground-truth CCC indices aligned with the QC-filtered AnnData."""
    kept_obs_indices = np.asarray(kept_obs_indices, dtype=np.int64)
    old_to_new = np.full(int(n_obs_before), -1, dtype=np.int64)
    old_to_new[kept_obs_indices] = np.arange(kept_obs_indices.size, dtype=np.int64)

    for gt_key in ("ground_truth", "cellvocal_ground_truth"):
        if gt_key not in adata.uns:
            continue
        gt = deepcopy(adata.uns[gt_key])

        events = gt.get("events") if isinstance(gt, dict) else None
        if isinstance(events, dict) and "sender_id" in events and "receiver_id" in events:
            sender_old = np.asarray(events["sender_id"], dtype=np.int64)
            receiver_old = np.asarray(events["receiver_id"], dtype=np.int64)
            n_events_before = sender_old.shape[0]
            valid_old = (
                (sender_old >= 0)
                & (sender_old < n_obs_before)
                & (receiver_old >= 0)
                & (receiver_old < n_obs_before)
            )

            sender_new = np.full(sender_old.shape, -1, dtype=np.int64)
            receiver_new = np.full(receiver_old.shape, -1, dtype=np.int64)
            sender_new[valid_old] = old_to_new[sender_old[valid_old]]
            receiver_new[valid_old] = old_to_new[receiver_old[valid_old]]
            keep_events = valid_old & (sender_new >= 0) & (receiver_new >= 0)

            remapped_events = {}
            for name, value in events.items():
                arr = np.asarray(value)
                if arr.shape[:1] == (n_events_before,):
                    arr = arr[keep_events]
                remapped_events[name] = arr
            remapped_events["sender_id"] = sender_new[keep_events]
            remapped_events["receiver_id"] = receiver_new[keep_events]
            gt["events"] = remapped_events
            gt["qc_event_remap"] = {
                "n_events_before": int(n_events_before),
                "n_events_after": int(keep_events.sum()),
                "n_events_removed_by_obs_qc": int(n_events_before - keep_events.sum()),
            }

        if isinstance(gt, dict) and "module_activation_prob" in gt:
            module_activation_prob = np.asarray(gt["module_activation_prob"])
            if module_activation_prob.shape[:1] == (n_obs_before,):
                gt["module_activation_prob"] = module_activation_prob[kept_obs_indices]

        if isinstance(gt, dict):
            gt["qc_obs_remap"] = {
                "n_obs_before": int(n_obs_before),
                "n_obs_after": int(kept_obs_indices.size),
            }
        adata.uns[gt_key] = gt


def apply_rna_velocity_qc(
    adata: ad.AnnData,
    *,
    unspliced_layer: str = "unspliced",
    spliced_layer: str = "spliced",
    spatial_key: str = "spatial",
    min_spliced_counts_per_spot: float = 1.0,
    min_unspliced_counts_per_spot: float = 1.0,
    min_genes_per_spot: int = 1,
    min_shared_counts_per_gene: float = 20.0,
    n_top_genes: int = 2000,
    normalize_per_spot: bool = False,
    copy: bool = True,
    uns_key: str = _QC_PREFIX,
) -> ad.AnnData:
    """
    Filter low-quality spots and low-information genes before RNA velocity training.

    The gene filter follows the scVelo-style ``min_shared_counts`` idea: a gene
    needs enough evidence in both spliced and unspliced layers, approximated by
    ``min(total_spliced, total_unspliced)``. The function records QC metrics in
    ``obs`` / ``var`` and a compact summary in ``uns[uns_key]``.
    """
    u = _as_layer(adata, unspliced_layer)
    s = _as_layer(adata, spliced_layer)
    if u.shape != s.shape or u.shape != adata.shape:
        raise ValueError(
            "Velocity layers must match AnnData shape: "
            f"adata={adata.shape}, {unspliced_layer}={u.shape}, {spliced_layer}={s.shape}"
        )
    if spatial_key not in adata.obsm:
        raise KeyError(f"AnnData must have obsm[{spatial_key!r}] for spatial RNA velocity QC")

    spliced_counts_spot = _sum_axis(s, axis=1)
    unspliced_counts_spot = _sum_axis(u, axis=1)
    velocity_genes_spot = _nnz_axis(s + u, axis=1)

    spliced_counts_gene = _sum_axis(s, axis=0)
    unspliced_counts_gene = _sum_axis(u, axis=0)
    shared_counts_gene = np.minimum(spliced_counts_gene, unspliced_counts_gene)
    spliced_cells_gene = _nnz_axis(s, axis=0)
    unspliced_cells_gene = _nnz_axis(u, axis=0)
    shared_cells_gene = _nnz_axis(s.multiply(u) if sparse.issparse(s) else (s > 0) & (u > 0), axis=0)

    spot_mask = (
        (spliced_counts_spot >= min_spliced_counts_per_spot)
        & (unspliced_counts_spot >= min_unspliced_counts_per_spot)
        & (velocity_genes_spot >= min_genes_per_spot)
        & _finite_mask(adata.obsm[spatial_key])
    )
    gene_mask = shared_counts_gene >= min_shared_counts_per_gene
    gene_mask &= _top_gene_mask_by_mean(spliced_counts_gene + unspliced_counts_gene, n_top_genes)

    if not spot_mask.any():
        raise ValueError("RNA velocity QC removed all spots; relax spot-level QC thresholds")
    if not gene_mask.any():
        raise ValueError("RNA velocity QC removed all genes; relax gene-level QC thresholds")

    out = adata.copy() if copy else adata
    out.obs[_prefixed("total_spliced_counts")] = spliced_counts_spot
    out.obs[_prefixed("total_unspliced_counts")] = unspliced_counts_spot
    out.obs[_prefixed("total_velocity_counts")] = spliced_counts_spot + unspliced_counts_spot
    out.obs[_prefixed("n_velocity_genes_by_counts")] = velocity_genes_spot
    out.var[_prefixed("spliced_counts")] = spliced_counts_gene
    out.var[_prefixed("unspliced_counts")] = unspliced_counts_gene
    out.var[_prefixed("shared_counts")] = shared_counts_gene
    out.var[_prefixed("spliced_cells")] = spliced_cells_gene
    out.var[_prefixed("unspliced_cells")] = unspliced_cells_gene
    out.var[_prefixed("shared_cells")] = shared_cells_gene

    kept_obs_indices = np.flatnonzero(spot_mask)
    out = out[spot_mask, gene_mask].copy()
    _remap_ground_truth_after_obs_filter(
        out,
        kept_obs_indices=kept_obs_indices,
        n_obs_before=adata.n_obs,
    )
    normalized_target = None
    if normalize_per_spot:
        normalized_target = _normalize_layers_by_spot(
            out,
            layers=(unspliced_layer, spliced_layer),
        )

    summary = {
        "n_obs_before": int(adata.n_obs),
        "n_vars_before": int(adata.n_vars),
        "n_obs_after": int(out.n_obs),
        "n_vars_after": int(out.n_vars),
        "n_obs_removed": int(adata.n_obs - out.n_obs),
        "n_vars_removed": int(adata.n_vars - out.n_vars),
        "kept_obs_indices": kept_obs_indices.astype(np.int64, copy=False),
        "unspliced_layer": unspliced_layer,
        "spliced_layer": spliced_layer,
        "spatial_key": spatial_key,
        "min_spliced_counts_per_spot": min_spliced_counts_per_spot,
        "min_unspliced_counts_per_spot": min_unspliced_counts_per_spot,
        "min_genes_per_spot": min_genes_per_spot,
        "min_shared_counts_per_gene": min_shared_counts_per_gene,
        "n_top_genes": n_top_genes,
        "normalize_per_spot": normalize_per_spot,
        "target_sum": normalized_target,
    }
    out.uns[uns_key] = summary
    print(
        f"RNA velocity QC: {out.n_obs}/{adata.n_obs} spots, "
        f"{out.n_vars}/{adata.n_vars} genes retained",
        flush=True,
    )
    return out
