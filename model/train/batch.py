"""Local batch construction for CellVOCAL train / inference."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from model.train.config import CellVOCALConfig
from model.train.data import slice_to_batch
from model.train.types import CCCEventIndex, SliceInput


def annotation_codes(adata, config: CellVOCALConfig) -> Optional[np.ndarray]:
    if not config.model.use_cell_annotation:
        return None
    for key in ("clusters", "cell_type", "domain"):
        if key in adata.obs:
            codes, _ = pd.factorize(adata.obs[key])
            return codes.astype(np.int64)
    return None


def index_events_by_receiver(events: CCCEventIndex, n_spots: int) -> List[np.ndarray]:
    recv = np.asarray(events.receiver, dtype=np.int64)
    if recv.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_spots)]
    if recv.min() < 0 or recv.max() >= n_spots:
        raise ValueError("CCC event receiver indices are outside the cell range")
    order = np.argsort(recv, kind="stable").astype(np.int64, copy=False)
    sorted_recv = recv[order]
    starts = np.searchsorted(sorted_recv, np.arange(n_spots), side="left")
    ends = np.searchsorted(sorted_recv, np.arange(n_spots), side="right")
    return [order[starts[i] : ends[i]] for i in range(n_spots)]


def local_edge_index(graph, local_cells: np.ndarray) -> np.ndarray:
    sub = graph[local_cells][:, local_cells].tocoo()
    return np.stack(
        [
            sub.row.astype(np.int64, copy=False),
            sub.col.astype(np.int64, copy=False),
        ],
        axis=0,
    )


def expand_graph_context(graph, base_cells: np.ndarray, gcn_num_layers: int) -> np.ndarray:
    base = np.unique(np.asarray(base_cells, dtype=np.int64))
    graph_csr = graph.tocsr()
    hops = max(0, int(gcn_num_layers))
    collected = [base]
    frontier = base
    for _ in range(hops):
        neigh = graph_csr[frontier].indices.astype(np.int64, copy=False)
        if neigh.size == 0:
            break
        current = np.unique(np.concatenate(collected))
        new_cells = np.setdiff1d(np.unique(neigh), current, assume_unique=False)
        if new_cells.size == 0:
            break
        collected.append(new_cells)
        frontier = new_cells
    return np.unique(np.concatenate(collected))


def make_local_batch(
    receiver_ids: np.ndarray,
    *,
    slice_in: SliceInput,
    events: CCCEventIndex,
    events_by_receiver: List[np.ndarray],
    graph,
    config: CellVOCALConfig,
    device: torch.device,
    annotation: Optional[np.ndarray] = None,
    event_semantics: Optional[np.ndarray] = None,
):
    receiver_ids = np.asarray(receiver_ids, dtype=np.int64)
    event_parts = [events_by_receiver[int(r)] for r in receiver_ids]
    event_idx = np.concatenate(event_parts).astype(np.int64, copy=False)

    context_cells = np.concatenate(
        [receiver_ids, events.sender[event_idx], events.receiver[event_idx]]
    )
    local_cells = expand_graph_context(
        graph,
        context_cells,
        config.model.gcn_num_layers,
    ).astype(np.int64, copy=False)
    target_local = np.searchsorted(local_cells, receiver_ids).astype(np.int64, copy=False)

    local_events = CCCEventIndex(
        sender=np.searchsorted(local_cells, events.sender[event_idx]).astype(np.int64, copy=False),
        receiver=np.searchsorted(local_cells, events.receiver[event_idx]).astype(
            np.int64, copy=False
        ),
        module=events.module[event_idx].astype(np.int64, copy=False),
        lr_compatibility=events.lr_compatibility[event_idx],
        candidate_score=events.candidate_score[event_idx],
        ligand_expr=None if events.ligand_expr is None else events.ligand_expr[event_idx],
        receptor_expr=None if events.receptor_expr is None else events.receptor_expr[event_idx],
    )
    local_slice = SliceInput(
        unspliced=slice_in.unspliced[local_cells],
        spliced=slice_in.spliced[local_cells],
        spatial=slice_in.spatial[local_cells],
        velocity_target_unspliced=(
            None
            if slice_in.velocity_target_unspliced is None
            else slice_in.velocity_target_unspliced[local_cells]
        ),
        velocity_target_spliced=(
            None
            if slice_in.velocity_target_spliced is None
            else slice_in.velocity_target_spliced[local_cells]
        ),
        time_prior=(
            None
            if slice_in.time_prior is None
            else slice_in.time_prior[local_cells]
        ),
        spot_ids=slice_in.spot_ids[local_cells],
        gene_names=slice_in.gene_names,
    )
    batch = slice_to_batch(
        local_slice,
        local_events,
        device=device,
        annotation=None if annotation is None else annotation[local_cells],
        event_semantics=None if event_semantics is None else event_semantics[event_idx],
        edge_index=local_edge_index(graph, local_cells),
    )
    return batch, event_idx, receiver_ids, target_local
