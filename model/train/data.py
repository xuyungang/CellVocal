"""Step 1: parse AnnData into slice input D = {U, S, spatial coords, C?, M?}."""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from model.train.types import BatchTensors, CCCEventIndex, CellVOCALResult, SliceInput

DEFAULT_LAYER_UNSPLICED = "unspliced"
DEFAULT_LAYER_SPLICED = "spliced"
DEFAULT_SPATIAL_KEY = "spatial"


def layer_to_dense_f32(layer) -> np.ndarray:
    """Convert an AnnData layer (dense or sparse) to a float32 ndarray."""
    if sparse.issparse(layer):
        return layer.toarray().astype(np.float32, copy=False)
    return np.asarray(layer, dtype=np.float32)


def validate_slice_anndata(
    adata: ad.AnnData,
    *,
    unspliced_layer: str = DEFAULT_LAYER_UNSPLICED,
    spliced_layer: str = DEFAULT_LAYER_SPLICED,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
) -> None:
    """Raise if required matrices / coordinates are missing."""
    missing = []
    if unspliced_layer not in adata.layers:
        missing.append(f"layers[{unspliced_layer!r}]")
    if spliced_layer not in adata.layers:
        missing.append(f"layers[{spliced_layer!r}]")
    if spatial_key not in adata.obsm:
        missing.append(f"obsm[{spatial_key!r}]")
    if missing:
        raise KeyError("CellVOCAL slice input incomplete: " + ", ".join(missing))


def ann_to_slice_input(
    adata: ad.AnnData,
    *,
    unspliced_layer: str = DEFAULT_LAYER_UNSPLICED,
    spliced_layer: str = DEFAULT_LAYER_SPLICED,
    spatial_key: str = DEFAULT_SPATIAL_KEY,
) -> SliceInput:
    """Extract numpy arrays for one spatial slice."""
    validate_slice_anndata(
        adata,
        unspliced_layer=unspliced_layer,
        spliced_layer=spliced_layer,
        spatial_key=spatial_key,
    )
    u = layer_to_dense_f32(adata.layers[unspliced_layer])
    s = layer_to_dense_f32(adata.layers[spliced_layer])
    return SliceInput(
        unspliced=u,
        spliced=s,
        spatial=np.asarray(adata.obsm[spatial_key], dtype=np.float64),
        spot_ids=np.asarray(adata.obs_names),
        gene_names=np.asarray(adata.var_names),
    )


def slice_to_batch(
    slice_in: SliceInput,
    events: CCCEventIndex,
    *,
    device: torch.device,
    annotation: Optional[np.ndarray] = None,
    event_semantics: Optional[np.ndarray] = None,
    edge_index: Optional[np.ndarray] = None,
) -> BatchTensors:
    """Collate slice arrays and candidate CCC events onto ``device`` for training / inference."""
    u = torch.as_tensor(slice_in.unspliced, device=device, dtype=torch.float32)
    s = torch.as_tensor(slice_in.spliced, device=device, dtype=torch.float32)
    spatial = torch.as_tensor(slice_in.spatial, device=device, dtype=torch.float32)
    return BatchTensors(
        unspliced=u,
        spliced=s,
        spatial=spatial,
        annotation=(
            None if annotation is None else torch.as_tensor(annotation, device=device)
        ),
        event_sender=torch.as_tensor(events.sender, device=device, dtype=torch.long),
        event_receiver=torch.as_tensor(events.receiver, device=device, dtype=torch.long),
        event_module=torch.as_tensor(events.module, device=device, dtype=torch.long),
        lr_compatibility=torch.as_tensor(
            events.lr_compatibility,
            device=device,
            dtype=torch.float32,
        ),
        candidate_score=torch.as_tensor(
            events.candidate_score,
            device=device,
            dtype=torch.float32,
        ),
        ligand_expr=(
            None
            if events.ligand_expr is None
            else torch.as_tensor(events.ligand_expr, device=device, dtype=torch.float32)
        ),
        receptor_expr=(
            None
            if events.receptor_expr is None
            else torch.as_tensor(events.receptor_expr, device=device, dtype=torch.float32)
        ),
        event_semantics=(
            None
            if event_semantics is None
            else torch.as_tensor(event_semantics, device=device, dtype=torch.float32)
        ),
        edge_index=(
            None
            if edge_index is None
            else torch.as_tensor(edge_index, device=device, dtype=torch.long)
        ),
        velocity_target_unspliced=(
            None
            if slice_in.velocity_target_unspliced is None
            else torch.as_tensor(
                slice_in.velocity_target_unspliced,
                device=device,
                dtype=torch.float32,
            )
        ),
        velocity_target_spliced=(
            None
            if slice_in.velocity_target_spliced is None
            else torch.as_tensor(
                slice_in.velocity_target_spliced,
                device=device,
                dtype=torch.float32,
            )
        ),
        time_prior=(
            None
            if slice_in.time_prior is None
            else torch.as_tensor(
                slice_in.time_prior,
                device=device,
                dtype=torch.float32,
            )
        ),
        n_spots=int(u.shape[0]),
        n_events=int(events.sender.shape[0]),
    )


def lr_pair_labels(lr_pairs: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Compact module labels for ``uns[{prefix}_ccc]['lr_pairs']``."""
    return np.asarray(
        [
            f"{p.get('ligand', f'L{idx}')}->{p.get('receptor', f'R{idx}')}"
            for idx, p in enumerate(lr_pairs)
        ],
        dtype=str,
    )


def lr_pair_module_ids(lr_pairs: Sequence[Mapping[str, str]]) -> np.ndarray:
    """Original module identifiers for compacted LR modules."""
    return np.asarray(
        [
            p.get("module_id", p.get("pathway", idx))
            for idx, p in enumerate(lr_pairs)
        ],
        dtype=str,
    )


def attach_results_to_anndata(
    adata: ad.AnnData,
    result: CellVOCALResult,
    *,
    prefix: str = "cv",
) -> ad.AnnData:
    """Write core CellVOCAL outputs into AnnData (minimal fields only)."""
    adata = adata.copy()
    adata.layers[f"{prefix}_velocity"] = result.velocity
    for suffix, value in (
        ("intrinsic_velocity", result.velocity_intrinsic),
        ("ccc_velocity", result.velocity_ccc),
    ):
        if value is not None:
            adata.layers[f"{prefix}_{suffix}"] = value
    adata.obs[f"{prefix}_time"] = result.latent_time
    adata.obsm[f"{prefix}_latent"] = result.latent_state
    if result.event_index is not None and result.ccc_event_influence is not None:
        ccc = {
            "sender": np.asarray(result.event_index.sender),
            "receiver": np.asarray(result.event_index.receiver),
            "module": np.asarray(result.event_index.module),
            "event_influence": np.asarray(result.ccc_event_influence),
            "candidate_score": np.asarray(result.event_index.candidate_score),
            "lr_compatibility": np.asarray(result.event_index.lr_compatibility),
        }
        if result.ccc_module_activation_prob is not None:
            ccc["module_activation_prob"] = np.asarray(result.ccc_module_activation_prob)
        if result.ccc_sender_share is not None:
            ccc["sender_share"] = np.asarray(result.ccc_sender_share)
        if result.event_index.ligand_expr is not None:
            ccc["ligand_expr"] = np.asarray(result.event_index.ligand_expr)
        if result.event_index.receptor_expr is not None:
            ccc["receptor_expr"] = np.asarray(result.event_index.receptor_expr)
        if result.lr_pair_labels is not None:
            ccc["lr_pairs"] = np.asarray(result.lr_pair_labels, dtype=str)
        if result.lr_pair_module_ids is not None:
            ccc["lr_pair_module_ids"] = np.asarray(result.lr_pair_module_ids, dtype=str)
        if result.event_semantics is not None:
            ccc["event_semantics"] = np.asarray(result.event_semantics)
        adata.uns[f"{prefix}_ccc"] = ccc
    return adata
