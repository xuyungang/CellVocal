"""Typed containers for CellVOCAL I/O (framework-level, torch-agnostic where possible)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class SliceInput:
    """Single spatial slice D = {U, S, X_sp}."""

    unspliced: np.ndarray  # (n_spots, n_genes)
    spliced: np.ndarray  # (n_spots, n_genes)
    spatial: np.ndarray  # (n_spots, d_sp)
    spot_ids: np.ndarray  # (n_spots,)
    gene_names: Optional[np.ndarray] = None
    velocity_target_unspliced: Optional[np.ndarray] = None
    velocity_target_spliced: Optional[np.ndarray] = None
    time_prior: Optional[np.ndarray] = None


@dataclass
class CCCEventIndex:
    """Candidate events e_ji^(m) = (j, i, m) from ``build_candidate_ccc_events``."""

    sender: np.ndarray  # (n_events,)
    receiver: np.ndarray  # (n_events,)
    module: np.ndarray  # (n_events,)
    lr_compatibility: np.ndarray  # rho_tilde_ji^(m), (n_events,)
    candidate_score: np.ndarray  # c_ji^(m), LR/spatial prior score, (n_events,)
    ligand_expr: Optional[np.ndarray] = None  # L_jm, (n_events,)
    receptor_expr: Optional[np.ndarray] = None  # R_im, (n_events,)


@dataclass
class CellVOCALResult:
    """Inference outputs written back to AnnData (minimal export surface)."""

    velocity: np.ndarray  # v^_i^RNA, (n_spots, n_genes)
    latent_time: np.ndarray  # tau^_i, (n_spots,)
    latent_state: np.ndarray  # z_i, (n_spots, latent_dim)
    velocity_intrinsic: Optional[np.ndarray] = None  # intrinsic ds/dtau, (n_spots, n_genes)
    velocity_ccc: Optional[np.ndarray] = None  # CCC-induced ds/dtau, (n_spots, n_genes)
    ccc_event_influence: Optional[np.ndarray] = None  # a_ji^(m), (n_events,)
    ccc_module_activation_prob: Optional[np.ndarray] = None  # module activation q_i^m, (n_events,)
    ccc_sender_share: Optional[np.ndarray] = None  # conditional sender share p_ji^m, (n_events,)
    event_index: Optional[CCCEventIndex] = None
    event_semantics: Optional[np.ndarray] = None  # (n_events, semantic_dim), translator only
    lr_pair_labels: Optional[np.ndarray] = None  # (n_modules,) "Lig->Rec" strings
    lr_pair_module_ids: Optional[np.ndarray] = None  # original module ids before pruning


@dataclass
class BatchTensors:
    """Collated torch batch for one spatial slice."""

    unspliced: Any = None
    spliced: Any = None
    spatial: Any = None
    annotation: Any = None
    latent: Any = None
    event_sender: Any = None
    event_receiver: Any = None
    event_module: Any = None
    lr_compatibility: Any = None
    candidate_score: Any = None
    ligand_expr: Any = None
    receptor_expr: Any = None
    event_semantics: Any = None
    edge_index: Any = None  # (2, num_edges) spatial graph for GCN encoder
    velocity_target_unspliced: Any = None
    velocity_target_spliced: Any = None
    time_prior: Any = None
    n_spots: int = 0
    n_events: int = 0
