"""Step 4: candidate CCC events e_ji^(m) within spatial neighbourhood."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse

from model.train.config import CellVOCALGraphConfig
from model.train.types import CCCEventIndex


def _gene_index_map(gene_names: np.ndarray) -> dict[str, int]:
    upper_to_idx: dict[str, int] = {}
    for i, g in enumerate(gene_names):
        key = str(g)
        upper_to_idx.setdefault(key.upper(), i)
        if key not in upper_to_idx:
            upper_to_idx[key] = i
    return upper_to_idx


def _resolve_gene_index(name: str, gene_to_idx: dict[str, int]) -> Optional[int]:
    name = str(name).strip()
    if not name:
        return None
    if name in gene_to_idx:
        return gene_to_idx[name]
    return gene_to_idx.get(name.upper())


def _pair_gene_names(pair: Mapping[str, Any], key: str) -> list[str]:
    genes = pair.get(f"{key}_genes")
    if genes is None:
        genes = [pair[key]]
    if isinstance(genes, str):
        genes = [genes]
    return [str(g).strip() for g in genes if str(g).strip()]


def _module_gene_indices(
    lr_pairs: Sequence[Mapping[str, Any]],
    gene_names: np.ndarray,
    modules: Sequence[int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    gene_to_idx = _gene_index_map(np.asarray(gene_names))
    ligand_idx: list[np.ndarray] = []
    receptor_idx: list[np.ndarray] = []
    for k, m in enumerate(modules):
        pair = lr_pairs[int(m)]
        lig = [
            _resolve_gene_index(g, gene_to_idx)
            for g in _pair_gene_names(pair, "ligand")
        ]
        rec = [
            _resolve_gene_index(g, gene_to_idx)
            for g in _pair_gene_names(pair, "receptor")
        ]
        if any(x is None for x in lig) or any(x is None for x in rec) or not lig or not rec:
            raise KeyError(
                f"Module {m} genes not in var_names: "
                f"ligand_genes={pair.get('ligand_genes', pair.get('ligand'))!r}, "
                f"receptor_genes={pair.get('receptor_genes', pair.get('receptor'))!r}"
            )
        ligand_idx.append(np.asarray(lig, dtype=np.int64))
        receptor_idx.append(np.asarray(rec, dtype=np.int64))
    return ligand_idx, receptor_idx


def _spatial_sender_receiver_edges(graph: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Rule 2: sender j ∈ N(i) from CSR row i (receiver)."""
    graph = graph.tocsr()
    receivers: list[np.ndarray] = []
    senders: list[np.ndarray] = []
    for i in range(graph.shape[0]):
        start, end = graph.indptr[i], graph.indptr[i + 1]
        neigh = graph.indices[start:end]
        if neigh.size == 0:
            continue
        receivers.append(np.full(neigh.shape[0], i, dtype=np.int64))
        senders.append(neigh.astype(np.int64, copy=False))
    if not senders:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(senders), np.concatenate(receivers)


def _complex_expr_for_cells(
    spliced: np.ndarray,
    cells: np.ndarray,
    gene_idx: np.ndarray,
) -> np.ndarray:
    vals = spliced[np.asarray(cells, dtype=np.int64)[:, None], gene_idx[None, :]]
    if hasattr(vals, "toarray"):
        vals = vals.toarray()
    vals = np.asarray(vals, dtype=np.float64)
    if vals.ndim == 1 or vals.shape[1] == 1:
        return vals.reshape(-1)
    return vals.min(axis=1)


def _module_lig_rec_expr(
    event_idx: np.ndarray,
    senders: np.ndarray,
    receivers: np.ndarray,
    spliced: np.ndarray,
    lig_genes: np.ndarray,
    rec_genes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _complex_expr_for_cells(spliced, senders[event_idx], lig_genes),
        _complex_expr_for_cells(spliced, receivers[event_idx], rec_genes),
    )


def _module_expr_thresholds(
    spliced: np.ndarray,
    gene_idx: Sequence[np.ndarray],
    *,
    delta: float,
    quantile: float,
) -> np.ndarray:
    """Per-module expression cutoffs for rule 1."""
    n_mod = len(gene_idx)
    thresholds = np.full(n_mod, delta, dtype=np.float64)
    q = float(quantile)
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"expr quantile must be in [0, 1], got {quantile}")
    cells = np.arange(spliced.shape[0], dtype=np.int64)
    for k in range(n_mod):
        vals = _complex_expr_for_cells(spliced, cells, gene_idx[k])
        thresholds[k] = max(delta, float(np.quantile(vals, q)))
    return thresholds


def _receiver_module_topk_mask(
    receivers: np.ndarray,
    modules: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> np.ndarray:
    """Keep top-k sender events for each receiver-module competition group."""
    if k <= 0:
        raise ValueError("receiver-module event budget must be positive")
    keep = np.zeros(receivers.shape[0], dtype=bool)
    groups = np.stack(
        [
            np.asarray(receivers, dtype=np.int64),
            np.asarray(modules, dtype=np.int64),
        ],
        axis=1,
    )
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    for g in range(inverse.max() + 1):
        idx = np.flatnonzero(inverse == g)
        if idx.size <= k:
            keep[idx] = True
        else:
            local_top = idx[np.argpartition(-scores[idx], k - 1)[:k]]
            keep[local_top] = True
    return keep


def _edge_distances(
    spatial_graph: sparse.csr_matrix,
    senders: np.ndarray,
    receivers: np.ndarray,
) -> np.ndarray:
    """Return stored edge distances when available, otherwise NaN."""
    graph = spatial_graph.tocsr()
    distances = np.full(senders.shape[0], np.nan, dtype=np.float64)
    for k, (j, i) in enumerate(zip(senders, receivers)):
        start, end = graph.indptr[int(i)], graph.indptr[int(i) + 1]
        cols = graph.indices[start:end]
        vals = graph.data[start:end]
        hit = np.flatnonzero(cols == int(j))
        if hit.size:
            distances[k] = float(vals[hit[0]])
    return distances


def _spatial_distance_weights(distances: np.ndarray) -> np.ndarray:
    """Convert stored graph distances to a local spatial kernel."""
    distances = np.asarray(distances, dtype=np.float64)
    finite = np.isfinite(distances)
    if not finite.any():
        return np.ones(distances.shape[0], dtype=np.float64)

    observed = distances[finite]
    if observed.size and np.allclose(observed, 1.0):
        return np.ones(distances.shape[0], dtype=np.float64)

    positive = observed[observed > 0.0]
    fallback_decay = float(np.median(positive) * 4.0) if positive.size else 1.0
    fallback_decay = max(fallback_decay, 1e-6)
    safe_distances = np.where(finite, distances, 0.0)
    return np.exp(-safe_distances / fallback_decay)


def _spatially_smoothed_spliced(spliced: np.ndarray, spatial_graph: sparse.csr_matrix) -> np.ndarray:
    """Denoise dropout-heavy LR expression with the same local graph used for candidates."""
    x = np.asarray(spliced, dtype=np.float64)
    graph = spatial_graph.tocsr().astype(np.float64)
    if graph.nnz == 0:
        return x
    weights = graph.copy()
    weights.data = _spatial_distance_weights(weights.data)
    weights = weights + sparse.eye(weights.shape[0], dtype=np.float64, format="csr")
    row_sum = np.asarray(weights.sum(axis=1)).ravel()
    row_sum[row_sum <= 0.0] = 1.0
    weights = sparse.diags(1.0 / row_sum).tocsr() @ weights
    return np.asarray(weights @ x, dtype=np.float64)


def _normalize_gene_expression(expr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize each gene so LR product is comparable across modules."""
    x = np.asarray(expr, dtype=np.float64)
    lo = np.nanmin(x, axis=0)
    hi = np.nanmax(x, axis=0)
    denom = hi - lo
    denom[denom < eps] = 1.0
    return np.clip((x - lo) / denom, 0.0, 1.0)


def build_candidate_ccc_events(
    spatial_graph: sparse.csr_matrix,
    n_modules: int,
    *,
    lr_pairs: Sequence[Mapping[str, Any]],
    spliced: np.ndarray,
    gene_names: np.ndarray,
    config: CellVOCALGraphConfig,
) -> CCCEventIndex:
    """
  4.6 Candidate event filtering.

  Rule 2 — spatial: sender j ∈ N(i) from ``spatial_graph``.
  Rules 1, 3–4 — require ``lr_pairs``, ``spliced``, and ``gene_names``:

  - Rule 1: L_jm > δ_L and R_im > δ_R (optional per-module quantile cutoffs);
    multi-gene ligand/receptor complexes use the minimum subunit expression.
  - Rule 3: prior score = normalized(L_jm) * normalized(R_im) * spatial_weight_ji.
  - Rule 4: top-B sender events per receiver-module group, ranked by the prior score.
    """
    if n_modules < 1:
        raise ValueError("n_modules must be >= 1")

    modules = list(range(n_modules))
    senders_base, receivers_base = _spatial_sender_receiver_edges(spatial_graph)
    edge_distance = _edge_distances(spatial_graph, senders_base, receivers_base)
    senders, receivers = senders_base, receivers_base
    if senders.size == 0:
        return CCCEventIndex(
            sender=np.empty(0, dtype=np.int64),
            receiver=np.empty(0, dtype=np.int64),
            module=np.empty(0, dtype=np.int64),
            lr_compatibility=np.empty(0, dtype=np.float32),
            candidate_score=np.empty(0, dtype=np.float32),
            ligand_expr=np.empty(0, dtype=np.float32),
            receptor_expr=np.empty(0, dtype=np.float32),
        )

    n_edges = senders.shape[0]
    n_mod = len(modules)
    senders = np.repeat(senders, n_mod)
    receivers = np.repeat(receivers, n_mod)
    module = np.tile(np.asarray(modules, dtype=np.int64), n_edges)
    spatial_weight = _spatial_distance_weights(np.repeat(edge_distance, n_mod))

    if len(lr_pairs) < max(modules) + 1:
        raise ValueError("lr_pairs length must cover all module indices")

    spliced = _normalize_gene_expression(
        _spatially_smoothed_spliced(np.asarray(spliced), spatial_graph)
    )
    ligand_idx, receptor_idx = _module_gene_indices(lr_pairs, np.asarray(gene_names), modules)
    mod_pos = {int(m): k for k, m in enumerate(modules)}

    lig_expr = np.zeros(module.shape[0], dtype=np.float64)
    rec_expr = np.zeros(module.shape[0], dtype=np.float64)
    module_event_idx = {
        int(m): np.flatnonzero(module == m) for m in np.unique(module)
    }
    n_workers = config.ccc_expr_workers
    if n_workers <= 1 or len(module_event_idx) <= 1:
        for m, event_idx in module_event_idx.items():
            pos = mod_pos[m]
            lig, rec = _module_lig_rec_expr(
                event_idx, senders, receivers, spliced, ligand_idx[pos], receptor_idx[pos]
            )
            lig_expr[event_idx] = lig
            rec_expr[event_idx] = rec
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                m: pool.submit(
                    _module_lig_rec_expr,
                    event_idx,
                    senders,
                    receivers,
                    spliced,
                    ligand_idx[mod_pos[m]],
                    receptor_idx[mod_pos[m]],
                )
                for m, event_idx in module_event_idx.items()
            }
            for m, fut in futures.items():
                event_idx = module_event_idx[m]
                lig, rec = fut.result()
                lig_expr[event_idx] = lig
                rec_expr[event_idx] = rec

    lig_thr = _module_expr_thresholds(
        spliced,
        ligand_idx,
        delta=config.ccc_delta_ligand,
        quantile=config.ccc_ligand_expr_quantile,
    )
    rec_thr = _module_expr_thresholds(
        spliced,
        receptor_idx,
        delta=config.ccc_delta_receptor,
        quantile=config.ccc_receptor_expr_quantile,
    )
    lig_thr_evt = lig_thr[[mod_pos[int(m)] for m in module]]
    rec_thr_evt = rec_thr[[mod_pos[int(m)] for m in module]]

    # Rule 1
    keep = (lig_expr > lig_thr_evt) & (rec_expr > rec_thr_evt)

    # Rule 3: raw LR compatibility and normalized LR/spatial prior for ranking.
    rho = lig_expr * rec_expr
    candidate_score = np.maximum(lig_expr, 0.0) * np.maximum(rec_expr, 0.0) * spatial_weight

    senders, receivers, module = senders[keep], receivers[keep], module[keep]
    lig_expr, rec_expr = lig_expr[keep], rec_expr[keep]
    rho = rho[keep]
    candidate_score = candidate_score[keep]

    if senders.size == 0:
        return CCCEventIndex(
            sender=senders,
            receiver=receivers,
            module=module,
            lr_compatibility=rho.astype(np.float32),
            candidate_score=candidate_score.astype(np.float32),
            ligand_expr=lig_expr.astype(np.float32),
            receptor_expr=rec_expr.astype(np.float32),
        )

    # Rule 4: receiver-module budget preserves sender competition inside each expert.
    keep = _receiver_module_topk_mask(
        receivers,
        module,
        candidate_score,
        config.ccc_event_budget_per_receiver,
    )
    senders, receivers, module, rho, candidate_score, lig_expr, rec_expr = (
        senders[keep],
        receivers[keep],
        module[keep],
        rho[keep],
        candidate_score[keep],
        lig_expr[keep],
        rec_expr[keep],
    )

    return CCCEventIndex(
        sender=senders,
        receiver=receivers,
        module=module,
        lr_compatibility=rho.astype(np.float32),
        candidate_score=candidate_score.astype(np.float32),
        ligand_expr=lig_expr.astype(np.float32),
        receptor_expr=rec_expr.astype(np.float32),
    )


def prune_lr_modules_by_events(
    lr_pairs: Sequence[Mapping[str, Any]],
    events: CCCEventIndex,
    *,
    min_events: int = 1,
) -> tuple[list[dict[str, Any]], CCCEventIndex]:
    """
    Drop LR modules with no surviving candidate events and compact module indices.

    Run after ``build_candidate_ccc_events`` so model/translator only see modules
    that actually participate in the filtered event set.
    """
    if min_events < 1:
        raise ValueError(f"min_events must be >= 1, got {min_events}")
    if events.sender.size == 0:
        raise ValueError("No candidate events remain after filtering; cannot prune modules.")

    pairs = [dict(p) for p in lr_pairs]
    module_ids = events.module.astype(np.int64, copy=False)
    counts = np.bincount(module_ids, minlength=max(int(module_ids.max()) + 1, len(pairs)))
    active = np.flatnonzero(counts >= min_events).astype(np.int64)
    if active.size == 0:
        raise ValueError(
            f"No LR modules remain after event-based pruning (min_events={min_events})."
        )

    keep_events = np.isin(module_ids, active)
    kept_module_ids = module_ids[keep_events]
    old_to_new = {int(old): new for new, old in enumerate(active.tolist())}
    filtered = [pairs[int(m)] for m in active.tolist()]
    new_module = np.array([old_to_new[int(m)] for m in kept_module_ids], dtype=np.int64)

    return filtered, CCCEventIndex(
        sender=events.sender[keep_events],
        receiver=events.receiver[keep_events],
        module=new_module,
        lr_compatibility=events.lr_compatibility[keep_events],
        candidate_score=events.candidate_score[keep_events],
        ligand_expr=(
            None if events.ligand_expr is None else events.ligand_expr[keep_events]
        ),
        receptor_expr=(
            None if events.receptor_expr is None else events.receptor_expr[keep_events]
        ),
    )
